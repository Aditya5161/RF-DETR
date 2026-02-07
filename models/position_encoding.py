import torch
import math

class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        b, c, h, w = x.shape
        y_embed = torch.arange(h, device=x.device).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, device=x.device).unsqueeze(0).repeat(h, 1)

        div_term = torch.exp(
            torch.arange(0, self.num_pos_feats, 2, device=x.device) *
            (-math.log(10000.0) / self.num_pos_feats)
        )

        pos_x = x_embed[..., None] * div_term
        pos_y = y_embed[..., None] * div_term

        pos = torch.cat([
            torch.sin(pos_x), torch.cos(pos_x),
            torch.sin(pos_y), torch.cos(pos_y)
        ], dim=-1)

        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return pos
