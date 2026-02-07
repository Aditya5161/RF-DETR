import torch
import torch.nn as nn
from .backbone import Backbone
from .transformer import Transformer
from .position_encoding import PositionalEncoding2D


class RFDETR(nn.Module):
    def __init__(self, num_classes, num_queries=100, hidden_dim=256):
        super().__init__()

        self.backbone = Backbone(hidden_dim)
        self.position_encoding = PositionalEncoding2D(hidden_dim // 2)
        self.transformer = Transformer(hidden_dim)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, images):
        features = self.backbone(images)
        pos = self.position_encoding(features)

        hs = self.transformer(
            features,
            self.query_embed.weight,
            pos
        )

        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid()

        return {
            "pred_logits": outputs_class.transpose(0, 1),
            "pred_boxes": outputs_bbox.transpose(0, 1)
        }
