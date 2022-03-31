from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.modules.deepvo import DeepVOInterface
from src.models.modules.deepvo.layers import FlowNetEncoder, VOPoseDecoder
from src.models.modules.deepvo.utils import weight_init
from src.utils.odometry_to_track import to_homogeneous_torch


class DeepVOReducedCNNFeatures(nn.Module, DeepVOInterface):
    def __init__(self, hidden_size: int, pose_decoder: VOPoseDecoder, pretrained_cnn_path: Optional[str] = None):
        super().__init__()

        self.encoder = FlowNetEncoder(6)  # Two RGBs
        self._cnn_feature_vector_size = self.encoder.conv6[0].out_channels

        self.lstm = nn.LSTM(
            input_size=self._cnn_feature_vector_size,
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

    def make_pairs(self, images: List[torch.Tensor]) -> torch.Tensor:
        tensor = torch.stack(images, dim=1)
        return torch.cat([tensor[:, :-1], tensor[:, 1:]], dim=2)

    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, C, H, W = pairs.size()
        c_in = pairs.view(batch_size * time_steps, C, H, W)

        encoded_images = self.encoder(c_in)
        _, C2, _, _ = encoded_images.size()
        return torch.mean(encoded_images, dim=[2, 3]).view(batch_size, time_steps, C2)

    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        return out

    def decode(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(seq)

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        coords, quats = self.decode(self.forward_seq(self.forward_cnn(self.make_pairs(images))))
        return to_homogeneous_torch(coords, quats)

    @property
    def cnn_feature_vector_size(self) -> int:
        return self._cnn_feature_vector_size
