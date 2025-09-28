from __future__ import annotations

import math
from abc import abstractmethod
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2.functional import pad


class Task(Enum):
    SEMANTIC = 0
    INSTANCE = 1
    PANOPTIC = 2


class InferenceBase(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        eval_topk_instances: int = 100,
        stuff_classes: list[int] | None = None,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.eval_topk_instances = eval_topk_instances
        self.stuff_classes = stuff_classes
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh

    @classmethod
    @abstractmethod
    def task(cls) -> Task:
        pass

    def scale_img_size_semantic(self, size: tuple[int, int]) -> list[int]:
        factor = max(self.img_size[0] / size[0], self.img_size[1] / size[1]).item()
        return [round(s.item() * factor) for s in size]

    def window_imgs_semantic(self, imgs: torch.Tensor) -> torch.Tensor:
        new_h, new_w = self.scale_img_size_semantic(imgs.shape[-2:])
        resized_imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear")

        num_crops = math.ceil(max(resized_imgs.shape[-2:]) / min(self.img_size))
        overlap = num_crops * min(self.img_size) - max(resized_imgs.shape[-2:])
        overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

        crops, origins = [], []
        for i in range(len(imgs)):
            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_imgs.shape[-2] > resized_imgs.shape[-1]:
                    crop = resized_imgs[i, :, start:end, :]
                else:
                    crop = resized_imgs[i, :, :, start:end]
                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins

    def revert_window_logits_semantic(
        self,
        crop_logits: torch.Tensor,
        origins: list,
        img_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return torch.stack(
            [
                F.interpolate(
                    (sums / counts)[None, ...], img_sizes[i], mode="bilinear"
                )[0]
                for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
            ]
        )

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor,
        class_logits: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=1)[..., :-1],
        )

    def scale_img_size_instance_panoptic(self, size: tuple[int, int]) -> list[int]:
        factor = min(self.img_size[0] / size[0], self.img_size[1] / size[1]).item()
        return [round(s.item() * factor) for s in size]

    def resize_and_pad_imgs_instance_panoptic(self, imgs: torch.Tensor) -> torch.Tensor:
        new_h, new_w = self.scale_img_size_instance_panoptic(imgs.shape[-2:])
        resized_imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear")

        pad_h = max(0, self.img_size[-2] - resized_imgs.shape[-2])
        pad_w = max(0, self.img_size[-1] - resized_imgs.shape[-1])
        padded_img = pad(resized_imgs, (0, pad_w, 0, pad_h))

        return padded_img

    def revert_resize_and_pad_logits_instance_panoptic(
        self,
        transformed_logits: torch.Tensor,
        img_sizes: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        logits = []
        for i in range(len(transformed_logits)):
            scaled_size = self.scale_img_size_instance_panoptic(img_sizes[i])
            logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
            logits_i = F.interpolate(
                logits_i[None, ...], img_sizes[i], mode="bilinear"
            )[0]
            logits.append(logits_i)

        return logits

    def to_per_pixel_preds_panoptic(
        self,
        mask_logits_list: list[torch.Tensor],
        class_logits: torch.Tensor,
        stuff_classes: list[int],
        mask_thresh: float,
        overlap_thresh: float,
    ) -> torch.Tensor:
        scores, classes = class_logits.softmax(dim=-1).max(-1)
        preds_list = []

        for i in range(len(mask_logits_list)):
            preds = -torch.ones(
                (*mask_logits_list[i].shape[-2:], 2),
                dtype=torch.int32,
                device=class_logits.device,
            )
            preds[:, :, 0] = self.num_classes

            keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
            if not keep.any():
                preds_list.append(preds)
                continue

            masks = mask_logits_list[i].sigmoid()
            segments = -torch.ones(
                *masks.shape[-2:],
                dtype=torch.int32,
                device=class_logits.device,
            )

            mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(dim=0)
            stuff_segment_ids, segment_id = {}, 0
            segment_and_class_ids = []

            for k, class_id in enumerate(classes[i][keep].tolist()):
                orig_mask = masks[keep][k] >= 0.5
                new_mask = mask_ids == k
                final_mask = orig_mask & new_mask

                orig_area = orig_mask.sum().item()
                new_area = new_mask.sum().item()
                final_area = final_mask.sum().item()
                if (
                    orig_area == 0
                    or new_area == 0
                    or final_area == 0
                    or new_area / orig_area < overlap_thresh
                ):
                    continue

                if class_id in stuff_classes:
                    if class_id in stuff_segment_ids:
                        segments[final_mask] = stuff_segment_ids[class_id]
                        continue
                    else:
                        stuff_segment_ids[class_id] = segment_id

                segments[final_mask] = segment_id
                segment_and_class_ids.append((segment_id, class_id))

                segment_id += 1

            for segment_id, class_id in segment_and_class_ids:
                segment_mask = segments == segment_id
                preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
                preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

            preds_list.append(preds)

        return torch.stack(preds_list)
