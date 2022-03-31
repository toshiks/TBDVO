from abc import ABC, abstractmethod

from src.datamodules.datasets.core.odometry_dataset import OdometryDataset


class OdometryDatasetFactory(ABC):
    @abstractmethod
    def __iter__(self) -> OdometryDataset:
        pass
