from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import InferenceBase


class MaskClassificationSemantic(InferenceBase):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
    ) -> None:
        super().__init__(network=network, img_size=img_size, num_classes=num_classes)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = self.network(crops)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], size=self.img_size, mode="bilinear"
        )

        crop_logits = self.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        return logits.argmax(dim=1)
