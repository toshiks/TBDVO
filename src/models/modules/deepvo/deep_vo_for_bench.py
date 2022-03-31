from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.modules.deepvo.utils import weight_init
from src.models.modules.deepvo.layers import VOPoseDecoder, FlowNetEncoder, PositionalEncoding


class DeepVOReducedCNNFeaturesS(nn.Module):
    def __init__(
            self, image_shape: Tuple[int, int], sequence_len: int,
            hidden_size: int, pose_decoder: VOPoseDecoder,
            pretrained_cnn_path: Optional[str] = None
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=int(1024),
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Dropout(),
            pose_decoder
        )

        self.apply(weight_init)

    def forward(self, pairs: torch.Tensor):
        out, _ = self.lstm(pairs)

        return self.decoder(out)


class DeepVOTransformerS(nn.Module):
    def __init__(
            self, image_shape: Tuple[int, int], sequence_len: int, hidden_size: int, pose_decoder: VOPoseDecoder,
            pretrained_cnn_path: Optional[str] = None
    ):
        super().__init__()
        self.transformer_cross = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=1024, nhead=8, dim_feedforward=hidden_size, batch_first=True, norm_first=True
            ),
            num_layers=2,
            norm=nn.LayerNorm(normalized_shape=1024)
        )

        self.position_encoding = PositionalEncoding(d_model=1024)

        self.decoder = nn.Sequential(
            nn.Dropout(),
            pose_decoder
        )

        self.apply(weight_init)

    def forward(self, pairs: torch.Tensor):
        out = self.transformer_cross(self.position_encoding(pairs))
        return self.decoder(out)


class DeepVOReducedCNNFeaturesF(nn.Module):
    def __init__(
            self, image_shape: Tuple[int, int], sequence_len: int,
            hidden_size: int, pose_decoder: VOPoseDecoder,
            pretrained_cnn_path: Optional[str] = None
    ):
        super().__init__()

        self.encoder = FlowNetEncoder(6)  # Two RGBs

        self.lstm = nn.LSTM(
            input_size=int(1024),
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Dropout(),
            pose_decoder
        )

        self.apply(weight_init)

        if pretrained_cnn_path is not None:
            self.encoder.load_pretrained(pretrained_cnn_path)

        self.value = nn.Parameter(torch.randn((1, sequence_len - 2, 1024)), requires_grad=False)

    def forward(self, pairs: torch.Tensor):
        batch_size, time_steps, C, H, W = pairs.size()
        c_in = pairs.view(batch_size * time_steps, C, H, W)
        encoded_images = self.encoder(c_in[-1:])

        _, C2, _, _ = encoded_images.size()
        encoded_images = torch.mean(encoded_images, dim=[2, 3]).view(1, 1, C2)
        encoded_images = torch.cat([self.value, encoded_images], dim=1)

        out, _ = self.lstm(encoded_images)

        return self.decoder(out)


class DeepVOTransformerF(nn.Module):
    def __init__(
            self, image_shape: Tuple[int, int], sequence_len: int, hidden_size: int, pose_decoder: VOPoseDecoder,
            pretrained_cnn_path: Optional[str] = None
    ):
        super().__init__()

        self.encoder = FlowNetEncoder(6)  # Two RGBs

        self.transformer_cross = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=1024, nhead=8, dim_feedforward=hidden_size, batch_first=True, norm_first=True
            ),
            num_layers=2,
            norm=nn.LayerNorm(normalized_shape=1024)
        )

        self.position_encoding = PositionalEncoding(d_model=1024)

        self.decoder = nn.Sequential(
            nn.Dropout(),
            pose_decoder
        )

        self.apply(weight_init)

        if pretrained_cnn_path is not None:
            self.encoder.load_pretrained(pretrained_cnn_path)

        self.value = nn.Parameter(torch.randn((1, sequence_len - 2, 1024)), requires_grad=False)

    def forward(self, pairs: torch.Tensor):
        batch_size, time_steps, C, H, W = pairs.size()
        c_in = pairs.view(batch_size * time_steps, C, H, W)
        encoded_images = self.encoder(c_in[-1:])

        _, C2, _, _ = encoded_images.size()
        encoded_images = torch.mean(encoded_images, dim=[2, 3]).view(1, 1, C2)
        encoded_images = torch.cat([self.value, encoded_images], dim=1)

        out = self.transformer_cross(self.position_encoding(encoded_images))

        return self.decoder(out)
