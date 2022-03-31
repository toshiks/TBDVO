import random
from typing import List, Tuple

import albumentations as A
import numpy as np


class HorizontalOdometryFlip:
    """Horizontal flipping of poses."""

    def __init__(self, always_apply=False, p=0.5):
        self._p = p
        self._always_apply = always_apply
        self._flip_mat = np.eye(4)
        self._flip_mat[0, 0] = -1
        self._first_pose = np.eye(4)
        self._first_pose_inv = np.linalg.inv(self._first_pose @ self._flip_mat)
        self._image_aug = A.HorizontalFlip(always_apply=True)

    def __call__(
            self, images: List[np.ndarray], odometries: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random.random() >= self._p and not self._always_apply:
            return images, odometries

        return ([self._image_aug(image=image)["image"] for image in images],
                [self._first_pose_inv @ (self._first_pose @ odometry) @ self._flip_mat for odometry in odometries])
