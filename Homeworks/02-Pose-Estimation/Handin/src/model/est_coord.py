from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis
import torch.nn.functional as F


class EstCoordNet(nn.Module):
    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.global_feat_pooling = nn.AdaptiveMaxPool1d(1)  # (B, 1024, 1)
        self.decoder = nn.Sequential(
            nn.Conv1d(1024 + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1),
        )

        self._init_weights()

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
        N = pc.shape[1]

        x = pc.transpose(1, 2)  # (B, 3, N)
        global_feat = self.encoder(x)  # (B, 1024, N)
        global_feat = self.global_feat_pooling(global_feat)  # (B, 1024, 1)
        global_feat = global_feat.expand(-1, -1, N)  # (B, 1024, N)
        global_feat = torch.cat([x, global_feat], dim=1)  # (B, 3+1024, N)

        pred_coord = self.decoder(global_feat)  # (B, 3, N)
        pred_coord = pred_coord.transpose(1, 2)  # (B, N, 3)
        coord_loss = F.mse_loss(pred_coord, coord)

        loss = coord_loss
        metric = dict(
            loss=loss,
            # additional metrics you want to log
        )
        return loss, metric

    @torch.no_grad()
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
        B, N, _ = pc.shape
        device = pc.device
        dtype = pc.dtype

        x = pc.transpose(1, 2)  # (B, 3, N)
        global_feat = self.encoder(x)  # (B, 1024, N)
        global_feat = self.global_feat_pooling(global_feat)  # (B, 1024, 1)
        global_feat = global_feat.expand(-1, -1, N)  # (B, 1024, N)
        global_feat = torch.cat([x, global_feat], dim=1)  # (B, 3+1024, N)

        pred_coord = self.decoder(global_feat)  # (B, 3, N)
        pred_coord = pred_coord.transpose(1, 2)  # (B, N, 3)

        # No RANSAC
        trans = torch.zeros(B, 3, device=device, dtype=dtype)
        rot = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        for b in range(B):
            pc_b = pc[b]
            pred_coord_b = pred_coord[b]
            R, t = self.estimate_transform_svd(pred_coord_b, pc_b)
            rot[b] = R
            trans[b] = t
        return trans, rot

        # RANSAC
        sample_time = 20
        num_points = 3
        threshold = 0.2**2

        trans = torch.zeros(B, 3, device=device, dtype=dtype)
        rot = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        rot[:] = torch.eye(3, device=device, dtype=dtype)

        for b in range(B):
            pc_b = pc[b]
            pred_coord_b = pred_coord[b]

            best_inlier_count = -1
            best_R = torch.eye(3, device=device, dtype=dtype)
            best_t = torch.zeros(3, device=device, dtype=dtype)
            best_inlier_indices = None

            for _ in range(sample_time):
                indices = np.random.choice(N, num_points, replace=False)
                sample_pred = pred_coord_b[indices]
                sample_pc = pc_b[indices]

                R_hyp, t_hyp = self.estimate_transform_svd(sample_pred, sample_pc)
                transformed_pred = pred_coord_b @ R_hyp.T + t_hyp

                distances = torch.sum((pc_b - transformed_pred) ** 2, dim=1)

                inliers = distances < threshold
                current_inlier_count = torch.sum(inliers).item()

                if current_inlier_count > best_inlier_count:
                    best_inlier_count = current_inlier_count
                    best_R = R_hyp
                    best_t = t_hyp
                    best_inlier_indices = torch.where(inliers)[0]

            if (
                best_inlier_indices is not None
                and len(best_inlier_indices) >= num_points
            ):
                inlier_pred = pred_coord_b[best_inlier_indices]
                inlier_pc = pc_b[best_inlier_indices]

                R_final, t_final = self.estimate_transform_svd(inlier_pred, inlier_pc)
                rot[b] = R_final
                trans[b] = t_final

            else:
                rot[b] = best_R
                trans[b] = best_t

        return trans, rot

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def estimate_transform_svd(
        src_points: torch.Tensor, dst_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate rigid transformation (R, t) such that dst_points ≈ R @ src_points + t
        using SVD.

        Parameters
        ----------
        src_points : torch.Tensor
            Source points, shape (M, 3)
        dst_points : torch.Tensor
            Destination points, shape (M, 3)

        Returns
        -------
        R : torch.Tensor
            Rotation matrix, shape (3, 3)
        t : torch.Tensor
            Translation vector, shape (3,)
        """
        assert src_points.shape[0] >= 3 and dst_points.shape[0] >= 3

        src_centroid = torch.mean(src_points, dim=0)
        dst_centroid = torch.mean(dst_points, dim=0)

        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid

        H = src_centered.T @ dst_centered  # (3, 3)

        U, S, Vh = torch.linalg.svd(H)
        V = Vh.T
        R = V @ U.T

        if torch.linalg.det(R) < 0:
            V[:, -1] *= -1
            R = V @ U.T

        t = dst_centroid - R @ src_centroid

        return R, t
