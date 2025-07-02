from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config


class EstPoseNet(nn.Module):
    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config

        self.rot_loss_weight = 1.0
        self.eps = 1e-7

        # Input: (B, 3, N)
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
        self.trans_decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.rot_decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 9),
        )
        # Output: (B, 12), 12 = 3 translation + 9 rotation matrix elements
        self._init_weights()

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        pred_trans, pred_rot = self.est(pc)
        trans_loss = F.mse_loss(pred_trans, trans)
        rot_loss = F.mse_loss(pred_rot, rot)
        rot_angle_loss = self._compute_geodesic_loss(pred_rot, rot)

        loss = trans_loss + self.rot_loss_weight * rot_loss

        metric = dict(
            loss=loss.item(),
            trans_loss=trans_loss.item(),
            rot_loss=rot_loss.item(),
            rot_angle_loss=rot_angle_loss.item(),
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
        """
        B = pc.shape[0]
        x = pc.transpose(1, 2)  # (B, 3, N)
        x = self.encoder(x)  # (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        x = x.view(B, -1)  # (B, 1024)
        pred_trans = self.trans_decoder(x)  # (B, 3)
        pred_rot = self.rot_decoder(x).view(B, 3, 3)  # (B, 3, 3)
        pred_rot = self._svd_orthogonalize(pred_rot)

        return pred_trans, pred_rot

    def _svd_orthogonalize(self, M: torch.Tensor) -> torch.Tensor:
        """
        Use SVD to orthogonalize a batch of 3x3 matrices.
        R = U * diag(1, 1, det(U @ V.T)) * V.T
        """
        U, _, Vh = torch.linalg.svd(M)

        S_eye = (
            torch.eye(3, device=M.device, dtype=M.dtype)
            .unsqueeze(0)
            .repeat(M.shape[0], 1, 1)
        )
        det_U = torch.det(U)
        det_Vh = torch.det(Vh)
        S_eye[:, 2, 2] = det_U * det_Vh
        R = U @ S_eye @ Vh

        return R

    def _compute_geodesic_loss(
        self, R_pred: torch.Tensor, R_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the geodesic distance (angle difference in radians) between two batches of rotation matrices.
        Loss = mean( arccos( (trace(R_pred^T @ R_gt) - 1) / 2 ) )
        """
        assert R_pred.shape == R_gt.shape and R_pred.shape[-2:] == (3, 3)
        R_diff = R_pred.mT @ R_gt
        trace = torch.einsum("bii->b", R_diff)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        angle_diff_rad = torch.acos(cos_theta)
        return torch.mean(angle_diff_rad)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
