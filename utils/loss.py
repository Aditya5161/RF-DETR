import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou
from .matcher import box_cxcywh_to_xyxy

class DETRLoss:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            device=src_logits.device
        )

        for i, j in zip(*idx):
            target_classes[i, j] = targets[i]["labels"][j]

        return F.cross_entropy(
            src_logits.flatten(0, 1),
            target_classes.flatten(0, 1)
        )

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes)
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )
        ).mean()

        return loss_bbox + loss_giou

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def __call__(self, outputs, targets, indices):
        return (
            self.loss_labels(outputs, targets, indices)
            + self.loss_boxes(outputs, targets, indices)
        )
