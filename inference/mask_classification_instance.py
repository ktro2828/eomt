from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import InferenceBase, Task


class MaskClassificationPanoptic(InferenceBase):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            stuff_classes=stuff_classes,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            **kwargs,
        )

    @classmethod
    def task(cls) -> Task:
        return Task.PANOPTIC

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_sizes = [img.shape[-2:] for img in imgs]
        x = imgs / 255.0
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(x)
        mask_logits_per_layer, class_logits_per_layer = self.network(transformed_imgs)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], size=self.img_size, mode="bilinear"
        )
        mask_logits_list = self.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes
        )

        preds = self.to_per_pixel_preds_panoptic(
            mask_logits_list,
            class_logits_per_layer[-1],
            self.stuff_classes,
            self.mask_thresh,
            self.overlap_thresh,
        )  # [1, H, W, 2]

        instance_preds = preds[..., 1].unsqueeze(0)  # [1, 1, H, W]

        return instance_preds
