from abc import ABC, abstractmethod
from typing import Mapping

import torch.nn as nn


class LossWithParameters(ABC, nn.Module):
    @property
    @abstractmethod
    def params(self) -> Mapping[str, nn.Parameter]:
        raise NotImplemented
