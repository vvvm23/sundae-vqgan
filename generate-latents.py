import logging

logger = logging.getLogger("PIL.PngImagePlugin")
logger.setLevel(logging.CRITICAL)
logger.disabled = True

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torchvision.transforms.functional as TF
from torchvision import transforms as T

import numpy as np
import argparse

import yaml
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from PIL import Image
from ptpt.utils import get_device, set_seed


def preprocess_vqgan(x):
    x = 2.0 * x - 1.0
    return x


def main(args):
    torch.set_grad_enabled(False)
    torch.inference_mode()

    seed = set_seed(args.seed)
    device = get_device(not args.no_cuda)

    cfg = OmegaConf.load(args.cfg_path)

    sd = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]

    vqgan = VQModel(
        **(
            cfg.model.params.first_stage_config.params
            if args.legacy_cfg
            else cfg.model.params
        )
    )

    if args.legacy_cfg:
        sd = {
            k.replace("first_stage_model.", ""): v
            for k, v in sd.items()
            if "first_stage_model." in k
        }

    vqgan.load_state_dict(sd)
    vqgan.to(device)
    vqgan.eval()

    target_size = (
        cfg.model.params.first_stage_config.params.ddconfig.resolution
        if args.legacy_cfg
        else cfg.model.params.ddconfig.resolution
    )
    dataset = ImageFolder(
        args.dataset_in,
        transform=T.Compose(
            [
                T.Resize(target_size),
                T.CenterCrop(
                    target_size
                ),  # should leave square images unaffected due to earlier resize
                T.ToTensor(),
            ]
        ),
    )

    if not args.latents_out:
        args.latents_out = str(Path(dataset_in)) + "-latents"

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.nb_workers
    )

    ldata_path = Path(args.latents_out)
    ldata_path.mkdir(parents=True, exist_ok=True)

    if args.use_class:
        nb_classes = len(dataset.classes)
        for c in range(nb_classes):
            (ldata_path / str(c)).mkdir(parents=True, exist_ok=True)

    count = 0
    for batch in tqdm(dataloader):
        for f in ["r"] if args.no_aug else ["r", "l"]:
            x, c = batch

            x = x.to(device)
            x = x if f == "r" else torch.flip(x, dims=[-1])
            x = preprocess_vqgan(x)

            *_, (*_, q) = vqgan.encode(x)
            q = q.view(x.shape[0], -1)

            q = q.cpu().numpy()

            for i, y in enumerate(q):
                if args.use_class:
                    np.save(
                        ldata_path
                        / str(c[i].item())
                        / f"{str(count+i).zfill(6)}.{f}.npy",
                        y,
                    )
                else:
                    np.save(ldata_path / f"{str(count+i).zfill(6)}.{f}.npy", y)

        count += batch[0].shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="config/vqgan/ffhq256.yaml")
    parser.add_argument("--ckpt-path", type=str, default="vqgan-ckpt/ffhq256.ckpt")
    parser.add_argument("--dataset-in", type=str, default="data/ffhq1024/")
    parser.add_argument("--latents-out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nb-workers", type=int, default=4)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--use-class", action="store_true")
    parser.add_argument("--legacy-cfg", action="store_true")
    args = parser.parse_args()

    main(args)
