import numpy as np
from .constants import PC_MIN, PC_MAX


def get_pc(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth image into point cloud using intrinsics

    All points with depth=0 are filtered out

    Parameters
    ----------
    depth: np.ndarray
        Depth image, shape (H, W)
    intrinsics: np.ndarray
        Intrinsics matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        Point cloud with shape (N, 3)
    """
    # Get image dimensions
    height, width = depth.shape
    # Create meshgrid for pixel coordinates
    v, u = np.meshgrid(range(height), range(width), indexing="ij")
    # Flatten the arrays
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    # Filter out invalid depth values
    valid = depth_flat > 0
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid]
    # Create homogeneous pixel coordinates
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    # Convert pixel coordinates to camera coordinates
    rays = np.linalg.inv(intrinsics) @ pixels
    # Scale rays by depth
    points = rays * depth_flat
    return points.T


def get_pc_world(depth, camer_pose, intrinsics: np.ndarray) -> np.ndarray:
    """
    Reconstructs a 3D point cloud in the world frame from an observation.
    """

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    H, W = depth.shape

    u_coords = np.arange(W)
    v_coords = np.arange(H)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    d_flat = depth.flatten()

    valid_depth_mask = d_flat > 0.001

    u_valid = u_flat[valid_depth_mask]
    v_valid = v_flat[valid_depth_mask]
    d_valid = d_flat[valid_depth_mask]

    if d_valid.size == 0:
        return np.empty((0, 3))  # No valid points

    x_c = (u_valid - cx) * d_valid / fx
    y_c = (v_valid - cy) * d_valid / fy
    z_c = d_valid
    points_camera_frame = np.vstack((x_c, y_c, z_c)).T  # (N, 3)

    # Transform points from camera frame to world frame
    points_camera_frame_homogeneous = np.hstack(
        (points_camera_frame, np.ones((points_camera_frame.shape[0], 1)))
    )
    points_world_frame_homogeneous = (camer_pose @ points_camera_frame_homogeneous.T).T
    points_world_frame = points_world_frame_homogeneous[:, :3]

    return points_world_frame


def calculate_table_height(pc, z_min=0.68, z_max=0.78):
    # 筛选出 z 值在范围内的点
    filtered_points = pc[
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] >= z_min)
        & (pc[:, 2] <= z_max)
    ]
    if len(filtered_points) == 0:
        np.save("full_pc_world_raw.npy", pc)
        raise ValueError("===筛选后没有点，检查输入或调整 z 值范围")

    # 计算 z 均值
    table_height = np.mean(filtered_points[:, 2])
    return table_height


def get_workspace_mask(pc: np.ndarray) -> np.ndarray:
    """Get the mask of the point cloud in the workspace."""
    table_height = calculate_table_height(pc)
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > table_height + 0.005)
        & (pc[:, 2] < PC_MAX[2])
    )
    # print(f"筛选后有 {np.sum(pc_mask)} 个点")
    return pc_mask
