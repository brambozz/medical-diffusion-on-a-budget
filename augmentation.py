# from flash import InputTransform
import albumentations as A
from dataclasses import dataclass
from pathlib import Path
from flash.image import ImageClassificationData
from typing import Callable, Tuple, Union
from torchvision import transforms as T
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.io.input_transform import InputTransform
import torch
import numpy as np
from torch import nn


class AlbumentationsAdapter(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        augmented_np = self.transform(image=np.moveaxis(x.numpy(), 0, -1))["image"]
        augmented_torch = torch.from_numpy(np.moveaxis(augmented_np, -1, 0))
        return augmented_torch


transforms = {}
transforms["light"] = A.Compose(
    [
        A.HorizontalFlip(p=0.2),
        A.GaussNoise(0.01, p=0.1),
        A.RandomGamma(p=0.2, gamma_limit=(80, 120)),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=25, p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.OpticalDistortion(p=0.2),
        A.ChannelDropout(p=0.2),
    ]
)
transforms["light"] = AlbumentationsAdapter(transforms["light"])
transforms["medium"] = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(0.05, p=0.5),
        A.RandomGamma(p=0.5, gamma_limit=(50, 100)),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.ChannelDropout(p=0.5),
    ]
)
transforms["medium"] = AlbumentationsAdapter(transforms["medium"])
transforms["heavy"] = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(0.1, p=1),
        A.RandomGamma(p=1, gamma_limit=(10, 100)),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=1),
        A.RandomBrightnessContrast(p=1),
        A.OpticalDistortion(p=1),
        A.ChannelDropout(p=0.8),
    ]
)
transforms["heavy"] = AlbumentationsAdapter(transforms["heavy"])


@dataclass
class ImageClassificationInputTransform(InputTransform):
    transform: Callable
    image_size: Tuple[int, int] = (196, 196)

    def per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    "input",
                    T.Compose(
                        [
                            T.ToTensor(),
                            T.Resize(self.image_size),
                        ]
                    ),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )

    def train_per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    "input",
                    T.Compose(
                        [
                            T.ToTensor(),
                            T.Resize(self.image_size),
                            self.transform,
                        ]
                    ),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )


transform_objects = {}
for level in ["light", "medium", "heavy"]:
    transform_objects[level] = ImageClassificationInputTransform(transforms[level])
