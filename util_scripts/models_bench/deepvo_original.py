from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.modules.deepvo import DeepVO
from src.models.modules.deepvo.layers import VOPoseDecoder


class DeepVOForBench(DeepVO):
    def __init__(
            self, image_shape: Tuple[int, int], sequence_len: int, hidden_size: int, pose_decoder: VOPoseDecoder,
            pretrained_cnn_path: Optional[str] = None
    ):
        super().__init__(image_shape, sequence_len, hidden_size, pose_decoder, pretrained_cnn_path)
        self._value = nn.Parameter(
            torch.randn((1, sequence_len - 2, self.cnn_feature_vector_size)), requires_grad=False
        )

    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, C, H, W = pairs.size()
        c_in = pairs.view(batch_size * time_steps, C, H, W)
        encoded_images = self.encoder(c_in[-1:])

        encoded_images = encoded_images.view(1, 1, -1)
        encoded_images = torch.cat([self._value, encoded_images], dim=1)
        return encoded_images
