import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
from torchvision import transforms as T

from PIL import Image
from torchvision.utils import save_image
from taming.models.vqgan import VQModel

import random
import time
import toml
import yaml
from math import sqrt
from types import SimpleNamespace
from pathlib import Path
from omegaconf import OmegaConf
from copy import deepcopy

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.callbacks import CallbackType
from ptpt.utils import set_seed, get_parameter_count, get_device

from x_transformers import TransformerWrapper, Encoder
from hourglass_transformer_pytorch import HourglassTransformerLM
from tqdm import tqdm

from einops import rearrange, repeat
from math import prod

from utils import preprocess_vqgan, decode_vqgan


def block_preprocess(img, block_size, block_position):
    mask = torch.ones(img.shape[0], *img.shape[2:])
    mask[
        :,
        block_position[0] : block_position[0] + block_size[0],
        block_position[1] : block_position[1] + block_size[1],
    ] = 0
    mask = mask.bool()

    img = img.clone()
    img[~repeat(mask, "n h w -> n c h w", c=img.shape[1])] = 0.0
    return img, mask


def random_preprocess(img, nb_blocks, block_size):
    mask = torch.ones(img.shape[0], *img.shape[2:])

    random_positions = random.sample(
        [
            (y, x)
            for x in range(img.shape[3] - block_size[1])
            for y in range(img.shape[2] - block_size[0])
        ],
        k=nb_blocks,
    )

    for pos in random_positions:
        mask[:, pos[0] : pos[0] + block_size[0], pos[1] : pos[1] + block_size[1]] = 0

    mask = mask.bool()

    img = img.clone()
    img[~repeat(mask, "n h w -> n c h w", c=img.shape[1])] = 0.0
    return img, mask


def pixel_to_q_mask(pixel_mask, f):
    mask = rearrange(pixel_mask, "n (h fh) (w fw) -> n h w (fh fw)", fh=f, fw=f)
    return torch.all(mask, dim=-1)


def main(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))

    args.steps = args.steps if args.steps else cfg.sampling["steps"]
    args.nb_samples = args.nb_samples if args.nb_samples else cfg.sampling["nb_samples"]
    args.temperature = (
        args.temperature if args.temperature else cfg.sampling["temperature"]
    )
    args.sample_proportion = (
        args.sample_proportion
        if args.sample_proportion
        else cfg.sampling["sample_proportion"]
    )
    args.end_temperature = (
        args.end_temperature if args.end_temperature else args.temperature
    )

    seed = set_seed(args.seed)
    info(f"random seed: {seed}")

    device = get_device(not args.no_cuda)
    vqgan_device = get_device(args.cuda_vqgan)

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
        )
    else:
        msg = f"unrecogized model type '{cfg.net['type']}'"
        error(msg)
        raise ValueError(msg)
    net = net.to(device)
    info(f"number of parameters: {get_parameter_count(net):,}")

    vqgan_cfg = OmegaConf.load(cfg.vqgan["config"])
    sd = torch.load(args.vqgan["checkpoint"], map_location="cpu")["state_dict"]

    vqgan = VQModel(
        **(
            vqgan_cfg.model.params.first_stage_config.params
            if args.legacy_cfg
            else vqgan_cfg.model.params
        )
    )
    vqgan = vqgan.to(vqgan_device)
    if args.legacy_cfg:
        sd = {
            k.replace("first_stage_model.", ""): v
            for k, v in sd.items()
            if "first_stage_model." in k
        }

    vqgan_dim = (
        vqgan_cfg.model.params.first_stage_config.params.embed_dim
        if args.legacy_cfg
        else vqgan_cfg.model.params.embed_dim
    )
    vqgan.load_state_dict(sd)

    if args.resume:
        net.load_state_dict(torch.load(args.resume)["net"])

    args.dataset_in = Path(args.dataset_in)

    if args.dataset_in.is_dir():
        dataset = ImageFolder(
            args.dataset_in,
            transform=T.Compose(
                [T.Resize(vqgan_cfg.data.params.validation.params.size), T.ToTensor()]
            ),
        )
        dataloader = DataLoader(dataset, batch_size=args.nb_samples, shuffle=True)
        original_img, _ = next(iter(dataloader))
    else:
        original_img = torch.from_numpy(np.array(Image.open(args.dataset_in))) / 255.0
        original_img = rearrange(original_img, "h w c -> 1 c h w")[:, :3]

    sample_dir = Path("inpaint")
    sample_dir.mkdir(exist_ok=True)

    net.eval()
    vqgan.eval()

    original_img = original_img.to(vqgan_device)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        x = preprocess_vqgan(original_img)
        *_, (*_, img_q) = vqgan.encode(x)
        embeddings = vqgan.quantize.embedding(img_q)
        embeddings = embeddings.view(
            original_img.shape[0], *cfg.vqgan["latent_shape"], vqgan_dim
        ).permute(0, 3, 1, 2)
        recon_img = vqgan.decode(embeddings)

    recon_img = torch.clamp(recon_img, -1.0, 1.0)
    recon_img = (recon_img + 1.0) / 2.0

    img_q = img_q.view(x.shape[0], -1).clone()

    corrupted_img, pixel_mask = random_preprocess(recon_img, 32, (128, 128))
    corrupted_img, pixel_mask = corrupted_img.to(
        device if args.cuda_vqgan else "cpu"
    ), pixel_mask.to(device)

    vqgan_f = int(
        2
        ** (
            len(
                vqgan_cfg.model.params.first_stage_config.params.ddconfig.ch_mult
                if args.legacy_cfg
                else vqgan_cfg.model.params.ddconfig.ch_mult
            )
            - 1
        )
    )
    vq_mask = pixel_to_q_mask(pixel_mask, f=vqgan_f)
    vq_mask = rearrange(vq_mask, "n h w -> n (h w)")
    img_q[~vq_mask] = torch.randint_like(
        img_q,
        vqgan_cfg.model.params.first_stage_config.params.n_embed
        if args.legacy_cfg
        else vqgan_cfg.model.params.n_embed,
    )[~vq_mask]
    img_q, vq_mask = img_q.to(device), vq_mask.to(device)

    save_id = int(time.time())
    for i in range(args.nb_inpaint):
        final_sample = net.sample(
            steps=args.steps,
            nb_samples=args.nb_samples,
            temperature=args.temperature,
            end_temperature=args.end_temperature,
            sample_proportion=args.sample_proportion,
            end_sample_proportion=args.end_sample_proportion,
            start_latent=img_q.clone(),
            inpaint_mask=vq_mask,
            status=not args.no_tqdm,
            early_stop=not args.no_early_stop,
            min_steps=args.min_steps,
            device=device,
        )

        final_sample = final_sample.to(vqgan_device)

        debug("decoding latents with VQ-GAN")

        img = decode_vqgan(final_sample, vqgan, (*cfg.vqgan["latent_shape"], vqgan_dim))

        debug("saving to file")
        save_image(
            img,
            sample_dir / f"inpaint-{save_id}.{i}.png",
            nrow=int(sqrt(args.nb_samples)),
        )

    save_image(
        corrupted_img,
        sample_dir / f"corrupted-{save_id}.png",
        nrow=int(sqrt(args.nb_samples)),
    )
    save_image(
        original_img,
        sample_dir / f"original-{save_id}.png",
        nrow=int(sqrt(args.nb_samples)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_in", type=str)
    parser.add_argument(
        "--cfg-path", type=str, default="config/ffhq256/ffhq256-hourglass.toml"
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--nb-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--end-temperature", type=float, default=None)
    parser.add_argument("--sample-proportion", type=float, default=None)
    parser.add_argument("--end-sample-proportion", type=float, default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--cuda-vqgan", action="store_true")
    parser.add_argument(
        "--inpaint-strategy",
        type=str,
        default="block",
        choices=["classic", "grid", "block", "random"],
    )
    parser.add_argument("--legacy-cfg", action="store_true")
    parser.add_argument("--nb-inpaint", type=int, default=1)
    args = parser.parse_args()

    main(args)
