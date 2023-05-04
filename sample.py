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

from ptpt.trainer import Trainer, TrainerConfig
from ptpt.log import debug, info, warning, error, critical
from ptpt.callbacks import CallbackType
from ptpt.utils import set_seed, get_parameter_count, get_device

from x_transformers import TransformerWrapper, Encoder
from hourglass_transformer_pytorch import HourglassTransformerLM
from tqdm import tqdm
from utils import decode_vqgan


def main(args):
    cfg = SimpleNamespace(**toml.load(args.cfg_path))

    args.steps = args.steps if args.steps else cfg.sampling["steps"]
    args.temperature = (
        args.temperature if args.temperature else cfg.sampling["temperature"]
    )
    args.end_temperature = (
        args.end_temperature if args.end_temperature else args.temperature
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

    sample_dir = Path(args.out_dir)
    sample_dir.mkdir(exist_ok=False)  # skip this one if it exists

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
        )
    else:
        msg = f"unrecogized model type '{cfg.net['type']}'"
        error(msg)
        raise ValueError(msg)
    net = net.to(device)
    info(f"number of parameters: {get_parameter_count(net):,}")

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
        net.load_state_dict(torch.load(args.resume)["net"])

    net.eval()

    count = 0
    pb = tqdm(total=args.nb_samples, disable=args.no_tqdm)
    while count < args.nb_samples:
        sample = net.sample(
            steps=args.steps,
            nb_samples=args.batch_size,
            temperature=args.temperature,
            end_temperature=args.end_temperature,
            sample_proportion=args.sample_proportion,
            end_sample_proportion=args.end_sample_proportion,
            min_steps=args.min_steps,
            early_stop=not args.no_early_stop,
            status=False,
            history=False,
            device=device,
        )
        if not args.cuda_vqgan:
            sample = sample.cpu()

        img = decode_vqgan(sample, vqgan, (*cfg.vqgan["latent_shape"], vqgan_dim))

        for sub_img in img:
            save_image(
                sub_img.unsqueeze(0),
                sample_dir / f"sample-{str(count+args.count_offset).zfill(6)}.png",
                padding=0,
            )
            count += 1
        pb.update(img.shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path", type=str, default="config/ffhq256/ffhq256-hourglass.toml"
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nb-samples", type=int, default=10_000)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--end-temperature", type=float, default=None)
    parser.add_argument("--sample-proportion", type=float, default=None)
    parser.add_argument("--end-sample-proportion", type=float, default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--cuda-vqgan", action="store_true")
    parser.add_argument("--out-dir", type=str, default="fid-samples")
    parser.add_argument("--legacy-cfg", action="store_true")
    parser.add_argument("--count-offset", type=int, default=0)
    args = parser.parse_args()

    main(args)
