import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Iterator

import numpy as np
from more_itertools import pairwise
from nibabel.quaternions import mat2quat, quat2mat
from nibabel.eulerangles import mat2euler, euler2mat
from torch.utils.data import Dataset

from src.datamodules.augs import OdometryTransform
from src.utils.quaternions_utils import extract_quaternion_with_yaw_only


class OdometryDataset(Dataset, ABC):
    """Class for loading abstract odometry dataset."""

    def __init__(
            self, dtype: np.dtype = np.float32,
            transform: Optional[OdometryTransform] = None,
            is_3dof: bool = False
    ):
        """Init abstract odometry dataset."""
        self._dtype = dtype
        self._transform = OdometryTransform() if transform is None else transform
        self._is_3dof = is_3dof

    def set_transform(self, transform: OdometryTransform):
        self._transform = transform

    @staticmethod
    def _homogeneous_pose_to_coord_quat(
            homogeneous: np.ndarray, dtype: np.dtype
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(homogeneous[:3, -1], dtype=dtype), np.array(mat2quat(homogeneous[:3, :3]), dtype=dtype)

    @staticmethod
    def _pose_6dof_to_3dof(
            homogeneous: np.ndarray, dtype: np.dtype
    ) -> np.ndarray:
        coord = np.array(homogeneous[:3, -1], dtype=dtype)
        angles = np.array(mat2quat(homogeneous[:3, :3]), dtype=dtype)

        angles = extract_quaternion_with_yaw_only(angles)
        coord[1] = 0

        homogeneous = np.eye(4, dtype=dtype)
        homogeneous[:3, -1] = coord
        homogeneous[:3, :3] = quat2mat(angles)

        return homogeneous

    @staticmethod
    def _homogeneous_track_to_coordinates_with_quaternions(
            homogeneous: List[np.ndarray], dtype
    ) -> Tuple[np.ndarray, np.ndarray]:
        coords = []
        quats = []

        for pose in homogeneous:
            coord, quat = OdometryDataset._homogeneous_pose_to_coord_quat(pose, dtype)

            coords.append(coord)
            quats.append(quat)

        return np.array(coords, dtype=dtype), np.array(quats, dtype=dtype)

    @property
    @abstractmethod
    def sequence_poses(self) -> List[np.ndarray]:
        raise NotImplemented

    @property
    def sequence_odometry(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [
            self._homogeneous_track_to_coordinates_with_quaternions(
                [np.dot(np.linalg.inv(prev), cur) for prev, cur in pairwise(sequence)],
                self._dtype
            )
            for sequence in self.sequence_poses
        ]

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplemented

    @abstractmethod
    def _get_first_pose(self) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def _get_poses_iterator(self) -> Iterator[np.ndarray]:
        raise NotImplemented

    @property
    def first_pose(self) -> np.ndarray:
        return self._get_first_pose()

    @property
    def poses(self) -> Iterator[np.ndarray]:
        return self._get_poses_iterator()
        # for pose in self._get_poses_iterator():
        #     yield self._homogeneous_pose_to_coord_quat(pose, self._dtype)

    @abstractmethod
    def _get_images_with_poses_from(self, index: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        raise NotImplemented

    def __getitem__(self, index) -> Dict:
        """Get subsequence by index."""
        poses, images = self._get_images_with_poses_from(index)
        if self._is_3dof:
            poses = [self._pose_6dof_to_3dof(pose, self._dtype) for pose in poses]

        odometry = [np.dot(np.linalg.inv(prev), cur) for prev, cur in pairwise(poses)]
        images, odometry = self._transform(images, odometry)
        # odometry = self._homogeneous_track_to_coordinates_with_quaternions(
        #     odometry,
        #     self._dtype
        # )
        odometry = np.array(odometry, dtype=np.float32)

        return {
            "odometry": odometry,
            "first_index": index,
            "track_name": self.name,
            "first_point": self.first_pose,
            "images": images
        }

    @abstractmethod
    def __len__(self):
        raise NotImplemented
