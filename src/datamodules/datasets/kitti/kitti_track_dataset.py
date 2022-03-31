from pathlib import Path

import cv2
import numpy as np
from albumentations import center_crop
from typing import Iterator, List, Optional, Tuple

from src.datamodules.augs import OdometryTransform
from src.datamodules.datasets.core import OdometryDataset
from src.datamodules.datasets.kitti.kitti_reader import Odometry
from src.datamodules.datasets.utils import subsequence_indices_from


class KittiTrackDataset(OdometryDataset):
    """Class for loading Kitti dataset."""

    def __init__(
            self, dataset_path: Path, seq_name: str, window_size: int,
            skip_prob: float = 0.0,
            transform: Optional[OdometryTransform] = None,
            dtype=np.float32,
            is_3dof: bool = False,
            image_size: Tuple[int, int] = (600, 180)
    ):
        """Init kitti dataset."""
        super().__init__(dtype=dtype, transform=transform, is_3dof=is_3dof)

        self._dataset_path = dataset_path
        self._seq_id = seq_name
        self._sequence_name = f"kitti_{seq_name}"
        self._window_size = window_size
        self._skip_prob = skip_prob
        self._image_size = image_size

        self._dataset = Odometry(str(self._dataset_path), self._seq_id)

    @staticmethod
    def _load_images(paths: List[str], image_size: Tuple[int, int]) -> List[np.ndarray]:
        image_list = []

        for image_path in paths:
            try:
                image = cv2.imread(image_path)
                image = center_crop(image, 360, 1200)
                image = cv2.resize(image, image_size)

                image_list.append(image)
            except Exception as exception:
                print(image_path)
                raise RuntimeError() from exception

        return image_list

    def _get_images_with_poses_from(self, index: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        indices = subsequence_indices_from(
            start_index=index, need_indices=self._window_size, seq_len=len(self._dataset.poses),
            skip_prob=self._skip_prob
        )

        images = self._load_images(self._dataset.cam2_files[indices], self._image_size)

        return self._dataset.poses[indices], images

    @property
    def sequence_poses(self) -> List[np.ndarray]:
        return [
            self._dataset.poses[subsequence_indices_from(i, self._window_size, len(self._dataset.poses), 0)]
            for i in range(self.__len__())
        ]

    @property
    def name(self) -> str:
        return self._sequence_name

    def _get_first_pose(self) -> np.ndarray:
        return self._dataset.poses[0]

    def _get_poses_iterator(self) -> Iterator[np.ndarray]:
        return iter(self._dataset.poses)

    def __len__(self):
        return len(self._dataset.poses) - self._window_size
