import struct
import numpy as np
from torchvision.utils import save_image
import yaml
import argparse
from argparse import Namespace




def save_imgs(imgs, to_size, name) -> None:
    # x = np.array(x)
    # x = np.transpose(x, (1, 2, 0)) * 255
    # x = x.astype(np.uint8)
    # imsave(name, x)

    # x = 0.5 * (x + 1)

    # to_size = (C, H, W)
    imgs = imgs.clamp(0, 1)
    imgs = imgs.view(imgs.size(0), *to_size)
    save_image(imgs, name)

def save_encoded(enc: np.ndarray, fname: str) -> None:
    enc = np.reshape(enc, -1)
    sz = str(len(enc)) + 'd'

    with open(fname, 'wb') as fp:
        fp.write(struct.pack(sz, *enc))

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, required=True)

    return parser.parse_args()


def get_config(args: argparse.Namespace) -> argparse.Namespace:
    with open(args.cfg, 'rt') as fp:
        cfg = Namespace(**yaml.safe_load(fp))
    return cfg


def dump_cfg(file: str, cfg: dict) -> None:
    fp = open(file, "wt")
    for k, v in cfg.items():
        fp.write("%15s: %s\n" % (k, v))
    fp.close()
