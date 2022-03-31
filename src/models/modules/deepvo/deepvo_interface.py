import abc
from typing import List, Tuple

import torch


class DeepVOInterface(abc.ABC):
    @abc.abstractmethod
    def make_pairs(self, images: List[torch.Tensor]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def decode(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @property
    @abc.abstractmethod
    def cnn_feature_vector_size(self) -> int:
        pass
