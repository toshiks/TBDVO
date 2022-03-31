from typing import Optional

import torch
import torch.nn as nn

from src.models.modules.deepvo import DeepVOTransformerEncoder
from src.models.modules.deepvo.layers import VOPoseDecoder


class DeepVOTransformerEncoderForBench(DeepVOTransformerEncoder):
    def __init__(
            self, sequence_len: int, hidden_size: int,
            pose_decoder: VOPoseDecoder, pretrained_cnn_path: Optional[str] = None
    ):
        super().__init__(hidden_size, pose_decoder, pretrained_cnn_path)
        self._value = nn.Parameter(
            torch.randn((1, sequence_len - 2, self.cnn_feature_vector_size)), requires_grad=False
        )

    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, C, H, W = pairs.size()
        c_in = pairs.view(batch_size * time_steps, C, H, W)
        encoded_images = self.encoder(c_in[-1:])

        _, C2, _, _ = encoded_images.size()
        encoded_images = torch.mean(encoded_images, dim=[2, 3]).view(1, 1, C2)
        encoded_images = torch.cat([self._value, encoded_images], dim=1)

        return encoded_images
