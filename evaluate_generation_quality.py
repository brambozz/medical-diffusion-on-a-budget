"""Evaluate quality of a generated set of images with respect to the
set used to train an embedding."""
import argparse
from pathlib import Path
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image
import numpy as np


def load_dir_as_tensor(dir_path):
    images = []
    for image_path in dir_path.rglob("*.png"):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)
        image = np.moveaxis(image, -1, 0)
        image = (image - np.mean(image)) / np.std(image)  # z-score normalize
        image = np.clip(image * 64 + 128, 0, 255).astype(
            np.uint8
        )  # set mean to 128, and 2std to 0/255
        images.append(image[None, ...])

    batch_tensor = torch.from_numpy(np.concatenate(images))
    return batch_tensor


def main(generated_path, reference_path):
    imgs_generated = load_dir_as_tensor(generated_path)
    imgs_reference = load_dir_as_tensor(reference_path)

    fid = FrechetInceptionDistance()
    fid.update(imgs_generated, real=False)
    fid.update(imgs_reference, real=True)
    fid_value = fid.compute().item()

    inception_score = InceptionScore()
    inception_score.update(imgs_generated)
    result, _ = inception_score.compute()
    is_value = result.item()

    return fid_value, is_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    fid_value, is_value = main(
        Path(args.generated_path),
        Path(args.reference_path),
    )
    print(f"FID: {fid_value}")
    print(f"IS: {is_value}")
