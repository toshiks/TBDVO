from typing import List

import numpy as np
import torch
from nibabel.eulerangles import euler2mat
from nibabel.quaternions import quat2mat


def conjugate(quaternion: torch.Tensor):
    """Conjugate of quaternion.

    Args:
        quaternion: tensor with dims (*, 4), where last is w, i, j, k of quaternion

    Returns:
        conjq : tensor with dims (*, 4), where last is w, i, j, k of conjugate of `q`

    """
    return quaternion * torch.tensor(  # pylint: disable=E1102
        [1.0, -1, -1, -1], dtype=quaternion.dtype, device=quaternion.device
    )


def mult(quaternion_left: torch.Tensor, quaternion_right: torch.Tensor):
    """Multiply two quaternions.

    See: https://en.wikipedia.org/wiki/Quaternions#Hamilton_product

    Args:
        quaternion_left: tensor with dims (N, 4), where last is w, i, j, k of quaternion
        quaternion_right: tensor with dims (N, 4), where last is w, i, j, k of quaternion

    Returns:
        quaternion_left * quaternion_right: tensor with dims (N, 4)

    """
    quaternion_right.to(quaternion_left.device)

    w_1 = quaternion_left[:, 0]
    x_1 = quaternion_left[:, 1]
    y_1 = quaternion_left[:, 2]
    z_1 = quaternion_left[:, 3]
    w_2 = quaternion_right[:, 0]
    x_2 = quaternion_right[:, 1]
    y_2 = quaternion_right[:, 2]
    z_2 = quaternion_right[:, 3]

    prod_w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
    prod_x = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
    prod_y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
    prod_z = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2
    return torch.stack([prod_w, prod_x, prod_y, prod_z], 1)


def quat2mat_torch(q: torch.Tensor):
    """ Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    return torch.tensor([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]]).to(q.device)


def rotate_vector(vector: torch.Tensor, quaternion: torch.Tensor):
    """Apply transformation in quaternion to vector.

    See: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Describing_rotations_with_quaternions

    Args:
        vector: tensor with dim (N, M, 3)
        quaternion: tensor with dim (N, M, 4)

    Returns:
        `vector` rotated by quaternion `quaternion`: tensor with dim (N, M, 3)

    """
    varr = torch.zeros((vector.size()[0], 4,), dtype=vector.dtype, device=vector.device)
    varr[:, 1:] = vector
    return mult(quaternion, mult(varr, conjugate(quaternion)))[:, 1:]


def odometry2track(homogeneous_poses: torch.Tensor, first_point=None):
    """Make track of poses from odometry.

        Args:
            homogeneous_poses: tensor with dim (N, M, 4, 4)
            first_point: coord and quats. maybe None

        Returns:
            poses: tensor with dim (N, M, 4, 4)

        """
    if first_point is None:
        track = [homogeneous_poses[:, 0, :]]
        start_index = 1
    else:
        track = [first_point]
        start_index = 0

    for i in range(start_index, homogeneous_poses.size()[1]):
        pose = homogeneous_poses[:, i]
        prev = track[-1]

        track.append(torch.matmul(prev, pose))

    return torch.stack(track, dim=1)


def odometry2track_q(coord_odometry: torch.Tensor, quat_odometry: torch.Tensor, first_point=None):
    """Make track of poses from odometry.

    Args:
        coord_odometry: tensor with dim (N, M, 3)
        quat_odometry: tensor with dim (N, M, 4)
        first_point: coord and quats. maybe None

    Returns:
        poses: Tuple[tensor with dim (N, M, 3), tensor with dim (N, M, 4)]

    """

    if first_point is None:
        coord_track = [coord_odometry[:, 0, :]]
        quat_track = [quat_odometry[:, 0, :]]
        start_index = 1
    else:
        coord_track = [first_point[0]]
        quat_track = [first_point[1]]
        start_index = 0

    for i in range(start_index, coord_odometry.size()[1]):
        quat_i = quat_odometry[:, i]
        coord_i = coord_odometry[:, i]

        prev_quat = quat_track[-1]
        prev_coord = coord_track[-1]

        new_coord = rotate_vector(coord_i, prev_quat) + prev_coord
        new_quat = mult(prev_quat, quat_i)
        coord_track.append(new_coord)
        quat_track.append(new_quat)

    return torch.stack(coord_track, dim=1), torch.stack(quat_track, dim=1)


def to_homogeneous_q(coords: np.ndarray, quats: np.ndarray) -> List[np.ndarray]:
    """Convert lists of coords and quaternions to matrix poses."""
    poses = []

    for i, coord in enumerate(coords):
        pose = np.eye(4)
        pose[:3, :3] = quat2mat(quats[i])
        pose[:3, -1] = coord
        poses.append(pose)

    return poses


def to_homogeneous_eulers(coords: np.ndarray, eulers: np.ndarray) -> List[np.ndarray]:
    """Convert lists of coords and eulers to matrix poses."""
    poses = []

    for i, coord in enumerate(coords):
        pose = np.eye(4)
        z, x, y = eulers[i]
        pose[:3, :3] = euler2mat(z, y, x)
        pose[:3, -1] = coord
        poses.append(pose)

    return poses


def compute_rotation_matrix_from_quaternion(quat):
    qw = quat[..., :1].contiguous()
    qx = quat[..., 1:2].contiguous()
    qy = quat[..., 2:3].contiguous()
    qz = quat[..., 3:].contiguous()

    # Unit quaternion rotation matrices computatation
    xx = 2. * qx * qx
    yy = 2. * qy * qy
    zz = 2. * qz * qz
    xy = 2. * qx * qy
    xz = 2. * qx * qz
    yz = 2. * qy * qz
    xw = 2. * qx * qw
    yw = 2. * qy * qw
    zw = 2. * qz * qw

    row0 = torch.cat((1 - yy - zz, xy - zw, xz + yw), -1).unsqueeze(-2)  # batch*1*3
    row1 = torch.cat((xy + zw, 1 - xx - zz, yz - xw), -1).unsqueeze(-2)  # batch*1*3
    row2 = torch.cat((xz - yw, yz + xw, 1 - xx - yy), -1).unsqueeze(-2)  # batch*1*3

    matrix = torch.cat([row0, row1, row2], dim=-2)

    return matrix


def to_homogeneous_torch(coords: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Convert lists of coords and quaternions to matrix poses."""
    poses = torch.zeros(coords.shape[0], coords.shape[1], 4, 4, dtype=coords.dtype, device=coords.device)

    poses[:, :, -1, -1] += 1
    poses[:, :, :3, -1] += coords
    poses[:, :, :3, :3] += compute_rotation_matrix_from_quaternion(quats)

    return poses
