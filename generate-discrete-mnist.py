import logging

logger = logging.getLogger("PIL.PngImagePlugin")
logger.setLevel(logging.CRITICAL)
logger.disabled = True

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, KMNIST


import torchvision.transforms.functional as TF
from torchvision import transforms as T

import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

from ptpt.utils import get_device, set_seed


def main(args):
    torch.set_grad_enabled(False)
    torch.inference_mode()

    seed = set_seed(args.seed)
    device = get_device(not args.no_cuda)

    transforms = T.Compose([T.ToTensor()])

    if args.dataset_name in ["mnist"]:
        train_dataset = MNIST("data/", train=True, download=True, transform=transforms)
        test_dataset = MNIST("data/", train=False, download=True, transform=transforms)
    elif args.dataset_name in ["kmnist"]:
        train_dataset = KMNIST("data/", train=True, download=True, transform=transforms)
        test_dataset = KMNIST("data/", train=False, download=True, transform=transforms)
    elif args.dataset_name in ["fashion", "fashionmnist"]:
        train_dataset = FashionMNIST(
            "data/", train=True, download=True, transform=transforms
        )
        test_dataset = FashionMNIST(
            "data/", train=False, download=True, transform=transforms
        )
    else:
        raise ValueError("Unrecognized dataset name!")

    data_out_path = Path(args.data_out)
    data_out_path.mkdir()
    (data_out_path / "train").mkdir()

    for c in range(10):
        (data_out_path / "train" / str(c)).mkdir()

    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.nb_workers
    )
    count = 0
    for batch in tqdm(dataloader):
        x, c = batch
        x = (x.to(device) * 255).long().cpu().numpy()
        for i, y in enumerate(x):
            np.save(
                data_out_path
                / "train"
                / str(c[i].item())
                / f"{str(count+i).zfill(6)}.npy",
                y,
            )
        count += batch[0].shape[0]

    (data_out_path / "eval").mkdir()

    for c in range(10):
        (data_out_path / "eval" / str(c)).mkdir()

    dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.nb_workers
    )
    count = 0
    for batch in tqdm(dataloader):
        x, c = batch
        x = (x.to(device) * 255).long().cpu().numpy()
        for i, y in enumerate(x):
            np.save(
                data_out_path
                / "eval"
                / str(c[i].item())
                / f"{str(count+i).zfill(6)}.npy",
                y,
            )
        count += batch[0].shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("data_out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nb-workers", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    main(args)
