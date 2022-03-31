import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from typing import List, Mapping, Optional

from src.datamodules.augs import HorizontalOdometryFlip, OdometryTransform
from src.datamodules.datasets.core.odometry_dataset import OdometryDataset


def _get_to_tensor_augmentations() -> A.Compose:
    return A.Compose([
        A.ToFloat(255.0, always_apply=True),
        ToTensorV2(),
    ])


def _get_train_augmentations() -> A.Compose:
    to_tensor_transform = _get_to_tensor_augmentations()
    return A.Compose([
        A.RGBShift(),
        A.CLAHE(),
        A.OneOf([
            A.RandomGamma(),
            A.RandomBrightnessContrast()
        ], p=1),
        A.ColorJitter(),
        to_tensor_transform
    ])


class KittiDatamodule(LightningDataModule):
    """Datamodule for kitti dataset."""

    def __init__(
            self,
            datasets: Mapping[str, Mapping[str, OdometryDataset]],
            num_workers: int = 4,
            batch_size: int = 8,
    ):
        """Create datamodule for kitti dataset.

        Args:
            datasets: dict config for each stage of training
            dataset_path: path to root folder of kitti dataset.
            num_workers: number of workers for train dataloader. For validation dataloader it's 3x.
            batch_size: batch size for train loop. For validation loop it's 6x.
        """
        super().__init__()

        self._val_factories = datasets['val']
        self._train_factories = datasets['train']

        self._num_workers = num_workers
        self._batch_size = batch_size

        self.train_dataset: Optional[ConcatDataset] = None
        self.val_datasets: Optional[List[OdometryDataset]] = None

    def setup(self, stage: Optional[str] = None):
        datasets = []
        for factory in self._train_factories.values():
            for dataset in factory:
                dataset.set_transform(
                    OdometryTransform(
                        image_only_transforms=_get_train_augmentations(),
                        image_pose_transforms=[HorizontalOdometryFlip(p=0.5)]
                    )
                )
                datasets.append(dataset)

        self.train_dataset = ConcatDataset(datasets)

        self.val_datasets = []
        for factory in self._val_factories.values():
            for dataset in factory:
                dataset.set_transform(
                    OdometryTransform(image_only_transforms=_get_to_tensor_augmentations())
                )
                self.val_datasets.append(dataset)

    def prepare_data(self, *args, **kwargs):
        """Prepare training and validation datasets."""

    @staticmethod
    def _make_dataloader_for_dataset(dataset, batch_size: int, num_workers: int,
                                     shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        """Get dataloader for training."""
        return self._make_dataloader_for_dataset(
            self.train_dataset, self._batch_size, self._num_workers, shuffle=True
        )

    def val_dataloader(self) -> List[DataLoader]:
        """Get dataloader for validation."""
        return [
            self._make_dataloader_for_dataset(
                dataset, self._batch_size, self._num_workers, shuffle=False
            ) for dataset in self.val_datasets
        ]
