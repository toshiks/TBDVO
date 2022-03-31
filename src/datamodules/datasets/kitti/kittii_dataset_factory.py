from pathlib import Path

from typing import List, Tuple

from src.datamodules.datasets.core import OdometryDataset, OdometryDatasetFactory
from src.datamodules.datasets.kitti.kitti_track_dataset import KittiTrackDataset


class KittiDatasetFactory(OdometryDatasetFactory):
    def __init__(
            self,
            path: str,
            sequences_name: List[str],
            window_size: int = 2,
            skip_prob: float = 0.0,
            is_3dof: bool = False,
            image_size: Tuple[int, int] = (600, 180)
    ):
        super().__init__()
        self._path = Path(path)
        self._sequences_name = sequences_name
        self._window_size = window_size
        self._skip_prob = skip_prob
        self._is_3dof = is_3dof
        self._image_size = image_size

    def __iter__(self) -> OdometryDataset:
        for seq_name in self._sequences_name:
            yield KittiTrackDataset(
                self._path,
                seq_name,
                self._window_size,
                self._skip_prob,
                is_3dof=self._is_3dof,
                image_size=self._image_size,
            )
