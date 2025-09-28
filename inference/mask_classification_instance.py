from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import masks_to_boxes

from .base import InferenceBase, Task


class MaskClassificationInstance(InferenceBase):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        eval_topk_instances: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            **kwargs,
        )

    @classmethod
    def task(cls) -> Task:
        return Task.INSTANCE

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
        mask_logits = torch.stack(mask_logits_list)  # [B, N, H, W]
        class_logits = class_logits_per_layer[-1]  # [B, N, C]

        B, N, C = class_logits.shape

        # Get class probabilities (excluding background class)
        class_probs = class_logits.softmax(dim=-1)[:, :, :-1]  # [B, N, C-1]

        # Find top scoring class for each query
        max_scores, max_indices = class_probs.max(dim=-1)  # [B, N]

        # Select top-k instances per batch
        topk_scores, topk_query_indices = max_scores.topk(
            min(self.eval_topk_instances, N), dim=1, sorted=False
        )  # [B, K]

        # Gather corresponding masks and class probabilities
        batch_indices = torch.arange(B, device=class_logits.device)[:, None]  # [B, 1]
        selected_mask_logits = mask_logits[
            batch_indices, topk_query_indices
        ]  # [B, K, H, W]
        selected_class_probs = class_probs[
            batch_indices, topk_query_indices
        ]  # [B, K, C-1]

        # Convert masks to binary and calculate mask scores
        masks = selected_mask_logits > 0  # [B, K, H, W]
        mask_areas = masks.flatten(-2).sum(dim=-1)  # [B, K]
        mask_scores = selected_mask_logits.sigmoid().flatten(-2).sum(dim=-1) / (
            mask_areas + 1e-6
        )  # [B, K]

        # Final scores combining class confidence and mask quality
        final_scores = topk_scores * mask_scores  # [B, K]

        # Generate bounding boxes from masks
        boxes_list = []
        for b in range(B):
            batch_masks = masks[b]  # [K, H, W]
            K = batch_masks.shape[0]

            # Handle empty masks by creating zero boxes
            batch_boxes = torch.zeros(K, 4, device=masks.device, dtype=torch.float32)

            for k in range(K):
                mask = batch_masks[k]
                if mask.sum() > 0:  # Only process non-empty masks
                    try:
                        box = masks_to_boxes(mask.unsqueeze(0))[0]  # [4]
                        batch_boxes[k] = box
                    except RuntimeError:
                        # If masks_to_boxes still fails, keep zero box
                        pass

            # Add scores as 5th column
            batch_boxes = torch.cat(
                [batch_boxes, final_scores[b : b + 1].T], dim=-1
            )  # [K, 5]
            boxes_list.append(batch_boxes)

        boxes = torch.stack(boxes_list)  # [B, K, 5]

        # Pad class probabilities to include background class
        labels = torch.cat(
            [
                selected_class_probs,
                torch.zeros(
                    B, selected_class_probs.shape[1], 1, device=class_logits.device
                ),
            ],
            dim=-1,
        )  # [B, K, C]

        return boxes, labels, masks
