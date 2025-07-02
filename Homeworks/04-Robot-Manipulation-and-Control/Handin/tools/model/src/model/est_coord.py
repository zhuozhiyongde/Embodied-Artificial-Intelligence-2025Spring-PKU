from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):
    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        self.fc1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1024),
            nn.Flatten(),
        )
        self.fc4 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, kernel_size=1),
        )

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        pc = pc.transpose(1, 2)
        local1 = self.fc1(pc)
        local2 = self.fc2(local1)
        global_fea = self.fc3(local2)
        batch_size, _, num_points = local1.shape
        global_fea_expanded = global_fea.unsqueeze(2).expand(
            -1, -1, num_points
        )  # [B, 1024, N]
        combined = torch.cat([local1, local2, global_fea_expanded], dim=1)
        pred_coord = self.fc4(combined)
        pred_coord = pred_coord.transpose(1, 2)
        criterion = nn.MSELoss()
        loss = criterion(pred_coord, coord) * 100
        metric = dict(
            loss=loss,
            # additional metrics you want to log
        )

        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        before = pc
        N = pc.shape[1]
        pc = pc.transpose(1, 2)
        local1 = self.fc1(pc)
        local2 = self.fc2(local1)
        global_fea = self.fc3(local2)
        batch_size, _, num_points = local1.shape
        global_fea_expanded = global_fea.unsqueeze(2).expand(
            -1, -1, num_points
        )  # [B, 1024, N]
        combined = torch.cat([local1, local2, global_fea_expanded], dim=1)
        pred_coord = self.fc4(combined)  # (B,3,N)
        before_t = before.transpose(1, 2)  # B,3,N,camera frame
        pred_coord_mean_center = pred_coord - pred_coord.mean(dim=2, keepdim=True)
        before_t_mean_center = before_t - before_t.mean(dim=2, keepdim=True)  # B,3,N
        H = torch.matmul(
            before_t_mean_center, pred_coord_mean_center.transpose(1, 2)
        )  # B,3,3
        U, S, V_T = torch.linalg.svd(H)
        R_corr = (
            torch.eye(3, device=U.device, dtype=U.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )  # [B, 3, 3]
        # Check determinant of U * Vt

        det_UVt = torch.linalg.det(torch.matmul(U, V_T))  # [B]

        R_corr[:, 2, 2] = torch.sign(det_UVt)

        rot = torch.matmul(U, torch.matmul(R_corr, V_T))  # [B, 3, 3]

        trans = before_t.mean(dim=2, keepdim=True) - torch.matmul(
            rot, pred_coord.mean(dim=2, keepdim=True)
        )
        trans = trans.squeeze(2)

        """trans=torch.zeros(batch_size,3).to(pc.device)
        rot=torch.eye(3).unsqueeze(0).repeat(batch_size,1,1).to(pc.device)
        return(trans,rot)"""
        # return (trans, rot)  # 这里没有使用RANSAC
        num_iterations = 20  # RANSAC迭代次数
        sample_size = 4  # 每次采样的点数
        threshold = 0.00001 * 1024  # 内点阈值
        best_inliers = []

        random_index = torch.randint(
            0, num_points, size=(batch_size, num_iterations, sample_size)
        )  # B,20,4
        sampling_before = before.unsqueeze(1).repeat(
            1, num_iterations, 1, 1
        )  # B,20,N,3
        sampling_before = torch.gather(
            sampling_before, 2, random_index.unsqueeze(3).expand(-1, -1, -1, 3)
        )
        sampling_pred_coord = (
            pred_coord.transpose(1, 2).unsqueeze(1).repeat(1, num_iterations, 1, 1)
        )  # B,20,N,3
        sampling_pred_coord = torch.gather(
            sampling_pred_coord, 2, random_index.unsqueeze(3).expand(-1, -1, -1, 3)
        )

        sampling_before_mean = sampling_before.mean(dim=2, keepdim=True)  # B,20,1,3
        sampling_pred_coord_mean = sampling_pred_coord.mean(
            dim=2, keepdim=True
        )  # B,20,1,3
        sampling_before_centered = sampling_before - sampling_before_mean  # B,20,4,3
        sampling_pred_coord_centered = (
            sampling_pred_coord - sampling_pred_coord_mean
        )  # B,20,4,3

        sampling_H = torch.matmul(
            sampling_before_centered.transpose(2, 3), sampling_pred_coord_centered
        )  # B,20,3,3
        sampling_U, sampling_S, sampling_VT = torch.linalg.svd(sampling_H)  # B,20,3,3
        sampling_R_corr = (
            torch.eye(3, device=sampling_U.device, dtype=sampling_U.dtype)
            .unsqueeze(0)
            .repeat(batch_size, num_iterations, 1, 1)
        )  # [B,20,3,3]
        sampling_det_UVt = torch.linalg.det(
            torch.matmul(sampling_U, sampling_VT)
        )  # B,20
        sampling_R_corr[:, :, 2, 2] = torch.sign(sampling_det_UVt)
        sampling_rot = torch.matmul(
            sampling_U, torch.matmul(sampling_R_corr, sampling_VT)
        )  # [B,20,3,3]
        a = torch.matmul(
            sampling_rot, sampling_pred_coord_mean.transpose(-1, -2)
        )  # B,20,3,1

        sampling_trans = sampling_before_mean - a.transpose(-1, -2)  # B,20,1,3

        before_repeat = before.unsqueeze(1).repeat(
            1, num_iterations, 1, 1
        )  # [B, 20, N, 3]
        pred_coord_repeat = pred_coord.unsqueeze(1).repeat(
            1, num_iterations, 1, 1
        )  # [B, 20, 3, N]
        # after_pc是rot作用在pred_coord上的结果

        sampling_trans_repeat = sampling_trans.repeat(1, 1, num_points, 1).reshape(
            batch_size, num_iterations, 3, num_points
        )  # B,20,3,N
        after_pc = (
            torch.matmul(sampling_rot, pred_coord_repeat) + sampling_trans_repeat
        )  # B,20,3,N
        after_pc = after_pc.transpose(2, 3)  # B,20,N,3

        # 计算内点距离
        distances = torch.norm(before_repeat - after_pc, dim=3)  # B,20,N
        inliers_mask = distances < threshold  # B,20,N
        inliers_count = inliers_mask.sum(dim=2)  # B,20
        best_inliers = inliers_count.max(dim=1)[0]  # B

        inlier_index = inliers_count.argmax(dim=1)  # B

        batch_indices = torch.arange(batch_size)
        inliner_mask = inliers_mask[batch_indices, inlier_index]  # B x N
        pred_coord = pred_coord.transpose(1, 2)  # B,N,3
        mask_expanded = inliner_mask.unsqueeze(-1).expand(
            batch_size, -1, 3
        )  # B x N x 3
        valid_points_cam = before[mask_expanded].reshape(-1, 3)
        valid_points_obj = pred_coord[mask_expanded].reshape(-1, 3)
        result_cam = torch.zeros_like(before)  # (B, N, 3)
        num_valid_per_batch = inliner_mask.sum(dim=1)  # (B,)
        cumulative_valid = torch.cat(
            [torch.tensor([0], device=before.device), num_valid_per_batch.cumsum(dim=0)]
        )[:-1]

        # 构造全局索引，用于将有效点填入结果张量
        batch_indices = (
            torch.arange(batch_size).unsqueeze(1).expand(batch_size, N)
        )  # (B, N)
        point_indices = torch.arange(N).unsqueeze(0).expand(batch_size, N)  # (B, N)

        # 构造一个布尔掩码，表示哪些位置需要填充有效点
        fill_mask = point_indices < num_valid_per_batch.unsqueeze(1)  # (B, N)

        # 将有效点填入结果张量
        result_cam[fill_mask] = valid_points_cam  # (B, N, 3)

        result_obj = torch.zeros_like(before)  # (B, N, 3)
        result_obj[fill_mask] = valid_points_obj  # (B, N, 3)

        final_rot, final_trans = compute_rigid_transform_no_loop(
            result_cam, result_obj, fill_mask
        )
        return final_trans, final_rot


def compute_rigid_transform_no_loop(result_cam, result_obj, fill_mask):
    B, N, D = result_cam.shape  # B (batch size) is defined here

    # 提取有效点
    valid_points_cam = result_cam[fill_mask].view(-1, D)  # (total_valid_points, 3)
    valid_points_obj = result_obj[fill_mask].view(-1, D)  # (total_valid_points, 3)

    # 计算每个 batch 的有效点数量
    num_valid_per_batch = fill_mask.sum(dim=1)  # (B,)
    cumulative_valid = torch.cat(
        [torch.tensor([0], device=result_cam.device), num_valid_per_batch.cumsum(dim=0)]
    )[:-1]

    # 构造全局索引
    batch_indices = (
        torch.arange(B).unsqueeze(1).expand(B, N)[fill_mask]
    )  # (total_valid_points,)

    # 按 batch 分组计算中心点
    def scatter_add(x, indices, dim):
        target_shape = (B, *x.shape[1:])  # (B, D)
        target = torch.zeros(
            target_shape, dtype=x.dtype, device=x.device
        )  # 初始化目标张量
        indices_expanded = indices.unsqueeze(1).expand(-1, x.size(1))  # 扩展索引形状
        return target.scatter_add_(dim, indices_expanded, x)

    centroids_cam = scatter_add(
        valid_points_cam, batch_indices, 0
    ) / num_valid_per_batch.unsqueeze(1)
    centroids_obj = scatter_add(
        valid_points_obj, batch_indices, 0
    ) / num_valid_per_batch.unsqueeze(1)

    # 去中心化
    centered_cam = valid_points_cam - centroids_cam[batch_indices]
    centered_obj = valid_points_obj - centroids_obj[batch_indices]

    # 计算协方差矩阵 H = centered_obj^T * centered_cam
    def scatter_add_outer(A, tensor_B, indices):  # Renamed second argument to tensor_B
        # B from the outer scope is now accessible
        # Assuming A and tensor_B have shape (total_valid_points, D)
        # target_shape should be (Batch_size, D, D)
        target_shape = (B, A.size(1), tensor_B.size(1))  # Use outer B for batch size
        target = torch.zeros(
            target_shape, dtype=A.dtype, device=A.device
        )  # Initialize target tensor
        # Expand indices based on its own shape and target dimensions
        indices_expanded = indices.view(-1, 1, 1).expand(
            -1, A.size(1), tensor_B.size(1)
        )
        outer_product = A.unsqueeze(2) * tensor_B.unsqueeze(
            1
        )  # (total_valid_points, D, D)
        return target.scatter_add_(0, indices_expanded, outer_product)

    H = scatter_add_outer(
        centered_obj, centered_cam, batch_indices
    ) / num_valid_per_batch.view(-1, 1, 1)

    # 使用 SVD 分解计算旋转矩阵
    U, _, Vt = torch.linalg.svd(H)  # U: (B, 3, 3), Vt: (B, 3, 3)
    rotations = Vt.transpose(-1, -2) @ U.transpose(-1, -2)  # (B, 3, 3)

    # 检查反射情况（确保 det(R) > 0）
    reflections = torch.det(rotations) < 0  # (B,)
    Vt[reflections, -1] *= -1
    rotations[reflections] = Vt[reflections].transpose(-1, -2) @ U[
        reflections
    ].transpose(-1, -2)

    # 计算平移向量
    translations = centroids_cam - torch.einsum(
        "bij,bj->bi", rotations, centroids_obj
    )  # (B, 3)

    return rotations, translations
