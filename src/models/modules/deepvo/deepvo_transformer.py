from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.modules.deepvo import DeepVOInterface
from src.models.modules.deepvo.layers import FlowNetEncoder, PositionalEncoding, VOPoseDecoder
from src.models.modules.deepvo.utils import weight_init
from src.utils.odometry_to_track import to_homogeneous_torch


class DeepVOTransformerEncoder(nn.Module, DeepVOInterface):
    def __init__(self, hidden_size: int, pose_decoder: VOPoseDecoder, pretrained_cnn_path: Optional[str] = None):
        super().__init__()

        self.encoder = FlowNetEncoder(6)  # Two RGBs
        self._cnn_feature_vector_size = self.encoder.conv6[0].out_channels
        self.position_encoding = PositionalEncoding(d_model=self._cnn_feature_vector_size)

        self.transformer_cross = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self._cnn_feature_vector_size, nhead=8, dim_feedforward=hidden_size, batch_first=True,
                norm_first=True
            ),
            num_layers=2,
            norm=nn.LayerNorm(normalized_shape=self._cnn_feature_vector_size)
        )

        self.decoder = nn.Sequential(
            nn.Dropout(),
            pose_decoder
        )

        self.apply(weight_init)

        if pretrained_cnn_path is not None:
            self.encoder.load_pretrained(pretrained_cnn_path)

    def make_pairs(self, images: List[torch.Tensor]) -> torch.Tensor:
        tensor = torch.stack(images, dim=1)
        return torch.cat([tensor[:, :-1], tensor[:, 1:]], dim=2)

    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, C, H, W = pairs.size()
        c_in = pairs.view(batch_size * time_steps, C, H, W)
        encoded_images = self.encoder(c_in)

        bt_size, C2, _, _ = encoded_images.size()
        return torch.mean(encoded_images, dim=[2, 3]).view(batch_size, time_steps, C2)

    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        position_encoded_seq = self.position_encoding(seq)
        return self.transformer_cross(position_encoded_seq)

    def decode(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(seq)

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        coords, quats = self.decode(self.forward_seq(self.forward_cnn(self.make_pairs(images))))
        return to_homogeneous_torch(coords, quats)

    @property
    def cnn_feature_vector_size(self) -> int:
        return self._cnn_feature_vector_size
