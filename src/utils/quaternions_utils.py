import numpy as np
from nibabel.quaternions import rotate_vector


def project_on_plane(ort_on_plane: np.ndarray, vector: np.ndarray) -> np.ndarray:
    ort_on_plane = np.array(ort_on_plane)
    vector = np.array(vector)

    return vector - ((vector @ ort_on_plane) / np.linalg.norm(ort_on_plane) ** 2) * ort_on_plane


def quaternion_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1 = np.array(v1)
    v2 = np.array(v2)

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    q = np.zeros(4, dtype=np.float64)
    q[1:] = np.cross(v1, v2)
    q[0] = np.linalg.norm(v2) + v2 @ v1
    q /= np.linalg.norm(q)

    return q


def extract_quaternion_with_yaw_only(quat: np.ndarray) -> np.ndarray:
    xz_ort_on_plane = np.array([0, 1, 0], dtype=np.float64)
    forward_vec = np.array([0, 0, 1], dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)

    new_forward_vec = rotate_vector(forward_vec, quat)
    new_forward_vec = project_on_plane(xz_ort_on_plane, new_forward_vec)

    new_quat = quaternion_between_vectors(forward_vec, new_forward_vec)

    return new_quat.astype(np.float)