import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from PIL import Image
from torchvision.utils import save_image
from taming.models.vqgan import VQModel

import time
import toml
import yaml
from math import sqrt
from types import SimpleNamespace
from pathlib import Path
from omegaconf import OmegaConf
from copy import deepcopy
from einops import rearrange

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.callbacks import CallbackType
from ptpt.utils import set_seed, get_parameter_count, get_device

from x_transformers import TransformerWrapper, Encoder
from hourglass_transformer_pytorch import HourglassTransformerLM
from tqdm import tqdm

from utils import decode_vqgan


def input_args(args):
    menu_input = None

    while menu_input not in [1, 2, 9]:
        info("1. Sample")
        info("2. Repeat Last")
        info("9. Exit")
        try:
            menu_input = int(input("> "))
        except KeyboardInterrupt:
            return None
        except:
            error("invalid input!")

    if menu_input == 9:
        return None

    sample_arg_names = [
        "steps",
        "nb_samples",
        "temperature",
        "end_temperature",
        "sample_proportion",
        "end_sample_proportion",
        "min_steps",
        "history",
        "class_idx",
    ]
    sample_args = {n: args[n] for n in sample_arg_names}

    if menu_input == 2:
        return sample_args

    for arg_name in sample_arg_names:
        info(f"{arg_name}: [{sample_args[arg_name]}]")
        menu_input = input("> ")
        if menu_input:
            if menu_input in ["True", "False"]:
                menu_input = menu_input == "True"
            else:
                menu_input = float(menu_input) if "." in menu_input else int(menu_input)
            sample_args[arg_name] = menu_input

    return sample_args


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
    args.end_sample_proportion = (
        args.end_sample_proportion
        if args.end_sample_proportion
        else args.sample_proportion
    )
    args.end_temperature = (
        args.end_temperature if args.end_temperature else args.temperature
    )
    args.class_idx = 0
    args.conditional = cfg.net["conditional"]

    seed = set_seed(args.seed)
    info(f"random seed: {seed}")

    device = get_device(not args.no_cuda)

    cfg.net["conditional"] = (
        cfg.net["conditional"] if "conditional" in cfg.net else False
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

    if not args.mnist:
        vqgan_cfg = OmegaConf.load(cfg.vqgan["config"])
        sd = torch.load(cfg.vqgan["checkpoint"], map_location="cpu")["state_dict"]

        vqgan = VQModel(
            **(
                vqgan_cfg.model.params.first_stage_config.params
                if args.legacy_cfg
                else vqgan_cfg.model.params
            )
        )
        vqgan_dim = (
            vqgan_cfg.model.params.first_stage_config.params.embed_dim
            if args.legacy_cfg
            else vqgan_cfg.model.params.embed_dim
        )
        if args.cuda_vqgan:
            vqgan = vqgan.to(device)
        if args.legacy_cfg:
            sd = {
                k.replace("first_stage_model.", ""): v
                for k, v in sd.items()
                if "first_stage_model." in k
            }

        vqgan.load_state_dict(sd)

    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location="cpu")["net"])
    net = net.to(device)
    info(f"number of parameters: {get_parameter_count(net):,}")

    sample_dir = Path("samples")
    sample_dir.mkdir(exist_ok=True)

    net.eval()
    sample_args = vars(args)
    while True:
        sample_args = input_args(sample_args)
        if not sample_args:
            break

        sample_start_time = time.time()
        final_sample = net.sample(
            status=not args.no_tqdm,
            early_stop=not args.no_early_stop,
            device=device,
            c=torch.tensor(
                [int(sample_args["class_idx"])] * sample_args["nb_samples"]
            ).to(device)
            if args.conditional
            else None,
            **sample_args,
        )

        if sample_args["history"]:
            _, history = final_sample
            save_id = int(time.time())

            for i, final_sample in enumerate(tqdm(history)):
                if args.cuda_vqgan:
                    final_sample = final_sample.to(device)

                if args.mnist:
                    img = (
                        rearrange(final_sample, "n (h w) -> n 1 h w", h=28, w=28)
                        / 255.0
                    )
                else:
                    img = decode_vqgan(
                        final_sample, vqgan, (*cfg.vqgan["latent_shape"], vqgan_dim)
                    )

                save_image(
                    img,
                    sample_dir / f"sample-{save_id}-{str(i).zfill(4)}.png",
                    nrow=int(sqrt(sample_args["nb_samples"])),
                )
        else:
            if not args.cuda_vqgan:
                final_sample = final_sample.cpu()

            if args.mnist:
                img = rearrange(final_sample, "n (h w) -> n 1 h w", h=28, w=28) / 255.0
            else:
                img = decode_vqgan(
                    final_sample, vqgan, (*cfg.vqgan["latent_shape"], vqgan_dim)
                )

            debug(f"total time: {time.time() - sample_start_time:.3f} seconds")
            debug("saving to file")
            save_image(
                img,
                sample_dir / f"sample-{int(time.time())}.png",
                nrow=int(sqrt(sample_args["nb_samples"])),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--cuda-vqgan", action="store_true")
    parser.add_argument("--legacy-cfg", action="store_true")
    parser.add_argument("--mnist", action="store_true")
    args = parser.parse_args()

    main(args)
