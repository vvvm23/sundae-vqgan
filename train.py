import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from PIL import Image
from torchvision.utils import save_image, make_grid
from torchvision.datasets import DatasetFolder
from taming.models.vqgan import VQModel

import toml
from types import SimpleNamespace
from pathlib import Path
from omegaconf import OmegaConf
import yaml

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.callbacks import CallbackType
from ptpt.utils import set_seed, get_parameter_count, get_device
from ptpt.wandb import WandbConfig

from x_transformers import TransformerWrapper, Encoder
from hourglass_transformer_pytorch import HourglassTransformerLM

import wandb
from tqdm import tqdm
from time import time
from math import sqrt
from shutil import copyfile
from einops import rearrange

from utils import decode_vqgan


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, split_sizes, split="train"):
        super().__init__()
        if isinstance(root_path, str):
            root_path = Path(root_path)
        files = list(root_path.glob("*.npy"))
        assert len(files) == sum(
            split_sizes
        ), "sum of splits is not equal to number of files found!"

        if split == "train":
            self.files = files[: split_sizes[0]]
        elif split == "eval":
            self.files = files[split_sizes[0] : sum(split_sizes[:2])]
        elif split == "test":
            self.files = files[sum(split_sizes[:2]) :]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(self.files[idx]))


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def main(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))
    seed = set_seed(args.seed)
    info(f"random seed: {seed}")

    cfg.net["conditional"] = (
        cfg.net["conditional"] if "conditional" in cfg.net else False
    )
    if cfg.net["conditional"]:
        loader_fn = lambda p: torch.from_numpy(np.load(p))
        train_dataset = DatasetFolder(
            cfg.data["train_root"], loader=loader_fn, extensions=".npy"
        )
        eval_dataset = DatasetFolder(
            cfg.data["eval_root"], loader=loader_fn, extensions=".npy"
        )

        assert len(train_dataset.classes) == cfg.net["num_classes"]
    else:
        train_dataset = LatentDataset(
            cfg.data["root"], cfg.data["split_sizes"], split="train"
        )
        eval_dataset = LatentDataset(
            cfg.data["root"], cfg.data["split_sizes"], split="eval"
        )

    if cfg.net["type"] in ["transformer", "vanilla"]:
        net = TransformerWrapper(
            num_tokens=cfg.data["vocab_size"],
            max_seq_len=cfg.data["sequence_length"],
            attn_layers=Encoder(
                dim=cfg.net["dim"],
                depth=cfg.net["depth"],
                head=cfg.net["nb_heads"],
                use_scalenorm=cfg.net["use_scalenorm"],
                ff_glu=cfg.net["use_glu"],
                rotary_pos_emb=cfg.net["use_rotary"],
                attn_dropout=cfg.net["attn_dropout"],
                ff_dropout=cfg.net["ff_dropout"],
            ),
        )

    elif cfg.net["type"] in ["hourglass"]:
        net = HourglassTransformerLM(
            num_tokens=cfg.data["vocab_size"],
            dim=cfg.net["dim"],
            max_seq_len=cfg.data["sequence_length"],
            heads=cfg.net["nb_heads"],
            shorten_factor=cfg.net["shorten_factor"],
            depth=tuple(cfg.net["depth"]),
            causal=False,
            rotary_pos_emb=cfg.net["use_rotary"],
            attn_type="vanilla",
            updown_sample_type="linear",
            conditional=cfg.net["conditional"],
            num_classes=cfg.net["num_classes"] if cfg.net["conditional"] else 0,
        )
    else:
        msg = f"unrecogized model type '{cfg.net['type']}'"
        error(msg)
        raise ValueError(msg)

    info(f"number of parameters: {get_parameter_count(net):,}")

    def get_random_text(shape):
        return torch.randint(cfg.data["vocab_size"], shape)

    def corrupt_text(batched_text):
        corruption_prob_per_sequence = torch.rand((batched_text.shape[0], 1))
        rand = torch.rand(batched_text.shape)
        mask = (rand < corruption_prob_per_sequence).to(batched_text.device)

        random_text = get_random_text(batched_text.shape).to(batched_text.device)
        return mask * random_text + ~mask * batched_text

    def logits_fn(net, batched_text, c=None):
        samples = corrupt_text(batched_text)
        all_logits = []
        for _ in range(cfg.unroll_steps):
            logits = net(samples, c=c)
            samples = Categorical(logits=logits).sample().detach()
            all_logits.append(logits)
        final_logits = torch.cat(all_logits, axis=0)
        return final_logits

    def loss_fn(net, batch):
        if cfg.net["conditional"]:
            batched_text, c = batch
        else:
            batched_text, c = batch, None

        if args.mnist:
            batched_text = rearrange(batched_text, "n c h w -> n (c h w)")

        logits = logits_fn(net, batched_text, c=c)
        targets = batched_text.repeat(cfg.unroll_steps, 1)
        accuracy = (logits.argmax(dim=-1) == targets).sum() / targets.numel()
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
        return loss, accuracy * 100.0

    if not args.mnist:
        vqgan_cfg = OmegaConf.load(cfg.vqgan["config"])
        sd = torch.load(cfg.vqgan["checkpoint"], map_location="cpu")["state_dict"]
        if args.legacy_cfg:
            sd = {
                k.replace("first_stage_model.", ""): v
                for k, v in sd.items()
                if "first_stage_model." in k
            }

        vqgan = VQModel(
            **(
                vqgan_cfg.model.params.first_stage_config.params
                if args.legacy_cfg
                else vqgan_cfg.model.params
            )
        )
        vqgan.load_state_dict(sd)

    @torch.inference_mode()
    def sample_fn(
        net,
        steps,
        nb_samples,
        temperature,
        sample_proportion,
        end_temperature=None,
        end_sample_proportion=None,
        status=False,
        history=False,
        min_steps=10,
        early_stop=True,
    ):
        device = get_device(not args.no_cuda)
        sample_history = [] if history else None

        batched_text = get_random_text((nb_samples, cfg.data["sequence_length"])).to(
            device
        )
        sample_mask = torch.zeros(nb_samples).bool().to(device)

        c = None
        if cfg.net["conditional"]:
            c = torch.randint(0, cfg.net["num_classes"], (nb_samples,)).to(device)

        temperatures = torch.linspace(temperature, end_temperature, steps)
        proportions = torch.linspace(sample_proportion, end_sample_proportion, steps)
        pb = tqdm(total=steps, disable=not status)
        for n, (temperature, sample_proportion) in enumerate(
            zip(temperatures, proportions)
        ):
            iter_start_time = time()
            old_sample_mask = sample_mask.clone()
            logits = net(
                batched_text[~sample_mask], c=c[~sample_mask] if c != None else None
            )
            sample = Categorical(logits=logits / temperature).sample()

            mask = (torch.rand(sample.shape) > sample_proportion).to(
                batched_text.device
            )
            sample[mask] = batched_text[~sample_mask][mask]
            if n >= min_steps:
                sample_mask[~sample_mask] = torch.all(
                    (sample == batched_text[~sample_mask]).view(sample.shape[0], -1),
                    dim=-1,
                )

            if early_stop and torch.all(sample_mask).item():
                break
            batched_text[~old_sample_mask] = sample

            if history:
                sample_history.append(batched_text.cpu().clone())

            pb.set_description_str(
                f"T: {temperature:.3f}, p: {sample_proportion:.3f}, [{sample_mask.long().sum().item()} / {batched_text.shape[0]}]"
            )
            pb.update(1)

        debug(f"stopped sampling after {n+1} steps.")

        if history:
            return batched_text, sample_history

        return batched_text

    trainer_cfg = TrainerConfig(
        **cfg.trainer,
        nb_workers=args.nb_workers,
        use_cuda=not args.no_cuda,
        use_amp=not args.no_amp,
        save_outputs=not args.no_save,
    )

    wandb_cfg = WandbConfig(
        project="sundae-mnist" if args.mnist else "sundae-vqgan",
        entity="afmck",
        config={"cfg": cfg, "args": args},
    )

    trainer = Trainer(
        net=net,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        test_dataset=eval_dataset,
        cfg=trainer_cfg,
        wandb_cfg=wandb_cfg if args.wandb else None,
    )

    if not args.mnist:
        vqgan_dim = (
            vqgan_cfg.model.params.first_stage_config.params.embed_dim
            if args.legacy_cfg
            else vqgan_cfg.model.params.embed_dim
        )

    @torch.inference_mode()
    def callback_sample(trainer):
        samples = sample_fn(trainer.net, **cfg.sampling, status=False).cpu()

        if args.mnist:
            img = rearrange(samples, "n (h w) -> n 1 h w", h=28, w=28) / 255.0
        else:
            debug("decoding latents with VQ-GAN")
            img = decode_vqgan(samples, vqgan, (*cfg.vqgan["latent_shape"], vqgan_dim))

        debug("saving to file")
        save_image(
            img,
            trainer.directories["root"]
            / f"sample-{str(trainer.nb_updates).zfill(6)}.jpg",
            nrow=int(sqrt(samples.shape[0])),
        )

        if trainer.wandb:
            wandb_img = wandb.Image(make_grid(img, nrow=int(sqrt(samples.shape[0]))))
            trainer.wandb.log({"samples": wandb_img}, commit=False)

        trainer.net.train()

    if not args.no_save:
        trainer.register_callback(CallbackType.EvalEpoch, callback_sample)
        copyfile(args.cfg_path, trainer.directories["root"] / "config.toml")

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path", type=str, default="config/ffhq256/ffhq256-hourglass.toml"
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--nb-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--legacy-cfg", action="store_true")
    parser.add_argument("--mnist", action="store_true")
    args = parser.parse_args()

    main(args)
