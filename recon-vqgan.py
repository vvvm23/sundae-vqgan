import logging

logger = logging.getLogger("PIL.PngImagePlugin")
logger.setLevel(logging.CRITICAL)
logger.disabled = True

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torchvision.utils import save_image

import numpy as np
import argparse
import time
from PIL import Image

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

    save_dir = Path("recon")
    save_dir.mkdir(exist_ok=True)

    seed = set_seed(args.seed)
    device = get_device(not args.no_cuda)

    cfg = OmegaConf.load(args.cfg_path)

    print("loading VQ-GAN model")
    sd = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    if args.legacy_cfg:
        sd = {
            k.replace("first_stage_model.", ""): v
            for k, v in sd.items()
            if "first_stage_model." in k
        }
    vqgan = VQModel(
        **(
            cfg.model.params.first_stage_config.params
            if args.legacy_cfg
            else cfg.model.params
        )
    ).to(device)
    vqgan.load_state_dict(sd)
    vqgan.eval()

    img_size = (
        cfg.model.params.first_stage_config.params.ddconfig.resolution
        if args.legacy_cfg
        else cfg.model.params.ddconfig.resolution
    )

    args.img_path = Path(args.img_path)
    if args.img_path.is_dir():
        dataset = ImageFolder(
            args.img_path, transform=T.Compose([T.Resize(img_size), T.ToTensor()])
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        img, _ = next(iter(dataloader))
    else:
        print(f"loading image '{args.img_path}' ")
        img = Image.open(args.img_path).resize((img_size, img_size))
        img = (
            torch.tensor(np.array(img)).permute(2, 0, 1)[:3].unsqueeze(0).float()
            / 255.0
        )
    x = preprocess_vqgan(img).to(device)

    print("processing image with VQ-GAN")
    quant, _, (*_, q) = vqgan.encode(x)
    print(f"latent shape: {q.shape}")
    recon = vqgan.decode(quant)

    save_id = int(time.time())
    save_path = save_dir / f"{save_id}-recon.png"
    print(f"saving image to file '{save_path}'")
    save_image((recon + 1.0) / 2, save_path)

    save_path = save_dir / f"{save_id}-original.png"
    save_image(img, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("--cfg-path", type=str, default="config/vqgan/ffhq256.yaml")
    parser.add_argument("--ckpt-path", type=str, default="vqgan-ckpt/ffhq256.ckpt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--legacy-cfg", action="store_true")
    args = parser.parse_args()

    main(args)
