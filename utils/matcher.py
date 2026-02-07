import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou # pyright: ignore[reportMissingImports]

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h
    ], dim=-1)

class HungarianMatcher:
    def __call__(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            cost_class = -out_prob[b][:, targets[b]["labels"]]
            cost_bbox = torch.cdist(out_bbox[b], targets[b]["boxes"], p=1)

            cost_iou = -box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(targets[b]["boxes"])
            )

            cost = cost_class + cost_bbox + cost_iou
            i, j = linear_sum_assignment(cost.cpu())
            indices.append((torch.tensor(i), torch.tensor(j)))
        return indices
