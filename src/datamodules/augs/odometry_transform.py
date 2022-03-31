from typing import Any, List, Optional, Tuple

import albumentations as A
import numpy as np


class OdometryTransform:
    """Transformation for poses."""

    def __init__(self, image_only_transforms: Optional[A.Compose] = None, image_pose_transforms: Optional[List] = None):
        if image_only_transforms is None:
            self._image_only_transforms = None
        else:
            self._image_only_transforms = A.ReplayCompose([
                image_only_transforms
            ])

        self._image_pose_transforms = image_pose_transforms

    def _apply_image_only_transforms(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if self._image_only_transforms is None:
            return images

        first_aug = self._image_only_transforms(image=images[0])
        reply_data = first_aug['replay']
        images[0] = first_aug['image']
        for i in range(1, len(images)):
            images[i] = A.ReplayCompose.replay(reply_data, image=images[i])['image']

        return images

    def _apply_image_pose_transforms(
            self, images: List[np.ndarray], poses: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self._image_pose_transforms is None:
            return images, poses

        for transform in self._image_pose_transforms:
            images, poses = transform(images, poses)

        return images, poses

    def __call__(self, images: List[np.ndarray], poses: List[np.ndarray]) -> Tuple[List[Any], List[Any]]:
        images, poses = self._apply_image_pose_transforms(images, poses)
        images = self._apply_image_only_transforms(images)
        return images, poses
