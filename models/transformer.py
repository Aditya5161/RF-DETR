import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, num_layers=6):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, src, query_embed, pos):
        """
        src: (B, C, H, W)
        query_embed: (num_queries, C)
        pos: (B, C, H, W)
        """

        b, c, h, w = src.shape

        # (HW, B, C)
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        # Encoder
        memory = self.encoder(src + pos)

        # Prepare queries
        num_queries = query_embed.shape[0]  # noqa: F841
        query_embed = query_embed.unsqueeze(1).repeat(1, b, 1)  # (Q, B, C)
        tgt = torch.zeros_like(query_embed)

        # Decoder
        hs = self.decoder(tgt, memory)

        return hs
