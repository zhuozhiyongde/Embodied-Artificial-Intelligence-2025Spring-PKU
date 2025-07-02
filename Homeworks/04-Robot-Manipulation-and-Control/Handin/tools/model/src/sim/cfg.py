import numpy as np
from dataclasses import dataclass, field

from src.robot.cfg import RobotCfg


@dataclass
class MjRenderConfig:
    height: int = 480
    """camera image height"""
    width: int = 640
    """camera image width"""
    lookat: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """which point to look at"""
    distance: float = 2.5
    """the distance between camera and the lookat point"""
    azimuth: float = np.deg2rad(180)
    """azimuth angle in radian"""
    elevation: float = np.deg2rad(-45)
    """elevation angle in radian"""
    fovy: float = None
    """fov of the y-axis in degrees"""
    save_dir: str = None
    """where to save the rendered images"""

    @staticmethod
    def from_intrinsics_extrinsics(
        height: int,
        width: int,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> "MjRenderConfig":
        """Note that this will lose some information of intrinsics and extrinsics, but our camera in mujoco doesn't suffer from this"""
        assert intrinsics.shape == (3, 3), "Invalid intrinsics shape"
        assert extrinsics.shape == (4, 4), "Invalid extrinsics shape"

        right, down, forward = extrinsics[:3, :3].T
        distance = 1
        lookat = extrinsics[:3, 3] + forward * distance

        azimuth = np.arctan2(forward[1], forward[0])
        elevation = np.arctan2(forward[2], np.linalg.norm(forward[:2]))

        fovy = (
            2 * np.arctan(0.5 * height / intrinsics[1, 1]) * (180 / np.pi)
        )  # in degrees

        return MjRenderConfig(
            height=height,
            width=width,
            lookat=lookat,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            fovy=fovy,
        )


@dataclass
class MjSimConfig:
    robot_cfg: RobotCfg
    headless: int = False
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    realtime_sync: bool = False
    use_ground_plane: bool = True
    use_debug_robot: bool = False
    viewer_cfg: MjRenderConfig = field(default_factory=MjRenderConfig)

    renderer_cfg: MjRenderConfig = None
