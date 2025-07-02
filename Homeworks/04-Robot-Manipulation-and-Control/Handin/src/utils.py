import os
import random
from typing import Optional
import numpy as np

from transforms3d.quaternions import quat2mat

from .type import Grasp
from .constants import PC_MAX, PC_MIN

def to_pose(
    trans: Optional[np.ndarray] = None, rot: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert translation and rotation into a 4x4 pose matrix.

    Parameters
    ----------
    trans: Optional[np.ndarray]
        Translation vector, shape (3,).
    rot: Optional[np.ndarray]
        Rotation matrix, shape (3, 3).

    Returns
    -------
    np.ndarray
        4x4 pose matrix.
    """
    ret = np.eye(4)
    if trans is not None:
        ret[:3, 3] = trans
    if rot is not None:
        ret[:3, :3] = rot
    return ret

def transform_grasp_pose(
    grasp: Grasp,
    est_trans: np.ndarray,
    est_rot: np.ndarray,
    cam_trans: np.ndarray,
    cam_rot: np.ndarray,
) -> Grasp:
    """
    Transform grasp from the object frame into the robot frame

    Parameters
    ----------
    grasp: Grasp
        The grasp to be transformed.
    est_trans: np.ndarray
        Estimated translation vector in the camera frame.
    est_rot: np.ndarray
        Estimated rotation matrix in the camera frame.
    cam_trans: np.ndarray
        Camera's translation vector in the robot frame.
    cam_rot: np.ndarray
        Camera's rotation matrix in the robot frame.

    Returns
    -------
    Grasp
        The transformed grasp in the robot frame.
    """
    # raise NotImplementedError
    return Grasp(
        trans=cam_rot @ (est_rot @ grasp.trans + est_trans) + cam_trans,
        rot=cam_rot @ est_rot @ grasp.rot,
        width=grasp.width,
    )

def rand_rot_mat() -> np.ndarray:
    """
    Generate a random rotation matrix with shape (3, 3) uniformly.
    """
    while True:
        quat = np.random.randn(4)
        if np.linalg.norm(quat) > 1e-6:
            break
    quat /= np.linalg.norm(quat)
    return quat2mat(quat)


def theta_to_2d_rot(theta: float) -> np.ndarray:
    """
    Convert a 2D rotation angle into a rotation matrix.

    Parameters
    ----------
    theta : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The resulting 2D rotation matrix (2, 2).
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rot_dist(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    The relative rotation angle between two rotation matrices.

    Parameters
    ----------
    r1 : np.ndarray
        The first rotation matrix (3, 3).
    r2 : np.ndarray
        The second rotation matrix (3, 3).

    Returns
    -------
    float
        The relative rotation angle in radians.
    """
    return np.arccos(np.clip((np.trace(r1 @ r2.T) - 1) / 2, -1, 1))

