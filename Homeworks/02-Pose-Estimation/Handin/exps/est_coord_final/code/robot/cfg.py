import os
from copy import deepcopy
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class CameraCfg:
    width: int = 640
    """image width"""
    height: int = 480
    """image height"""
    intrinsics: np.ndarray = None
    """[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]"""
    extrinsics: np.ndarray = None
    """Relative pose matrix (4, 4) of camera to the specified link's frame."""
    link_name: str = None
    """link it is mounted on"""


@dataclass
class RobotCfg:
    path_urdf: str = None
    """path to urdf file"""
    path_mjcf: str = None
    """used for mujoco"""
    path_collision: str = None
    """contain collision spheres"""

    link_eef: str = None
    """end effector link name"""

    joint_names: List[str] = field(default_factory=lambda: [])
    """list of joints"""
    joint_init_qpos: np.ndarray = None
    """(J,) initial joint position"""
    joint_arm_indices: List[int] = field(default_factory=lambda: [])
    """the indices of arm joints"""
    joint_gripper_indices: List[int] = field(default_factory=lambda: [])
    """the indices of gripper joints"""

    gripper_width_to_angle_depth: Callable[[float], Tuple[float, float]] = None
    """find which gripper angle can lead to the specified width, and calculate the change of fingertip relative to eef (depth) since the fingertip moves forward when closing gripper"""

    collision_ignore: Dict[str, List[str]] = field(default_factory=dict)
    """link to a list of links"""

    camera_cfg: CameraCfg = None
    """camera configuration"""

    def is_collision_ignore(self, link1: str, link2: str) -> bool:
        if link1 in self.collision_ignore and link2 in self.collision_ignore[link1]:
            return True
        if link2 in self.collision_ignore and link1 in self.collision_ignore[link2]:
            return True
        return False


WRIST_CAMERA = CameraCfg(
    width=640,
    height=360,
    intrinsics=np.array([[326, 0, 320], [0, 326, 180], [0, 0, 1]]),
    extrinsics=np.array(
        [
            [0, 0, 1, 0],
            [0, -1, 0, 0.0655],
            [1, 0, 0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    link_name="left_arm_end_effector_mount_link",
)

LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
ROBOTIQ_JOINTS = [f"left_robotiq_left_joint_{i}" for i in range(3)] + [
    f"left_robotiq_right_joint_{i}" for i in range(3)
]

GALBOT_CHARLIE_COLLISION_IGNORE = {
    "ground": ["omni_chassis_base_link"],
    "omni_chassis_base_link": ["leg_link1", "leg_link2"],
    "leg_link1": ["leg_link2"],
    "leg_link2": ["torso_base_link"],
    "torso_base_link": [
        "left_arm_link2",
        "left_arm_link3",
        "right_arm_link2",
        "right_arm_link3",
        "head_link2",
    ],
    "left_arm_link2": ["left_arm_link3", "left_arm_link4", "left_arm_link5"],
    "left_arm_link3": ["left_arm_link4", "left_arm_link5"],
    "left_arm_link4": ["left_arm_link5", "left_arm_link6"],
    "left_arm_link5": ["left_arm_link6", "left_arm_link7"],
    "left_arm_link6": ["left_arm_link7", "robotiq_arg2f_base_link"],
    "left_arm_link7": ["robotiq_arg2f_base_link"],
    "robotiq_arg2f_base_link": [
        "left_outer_finger",
        "left_inner_finger",
        "left_inner_knuckle",
        "left_outer_knuckle",
        "right_outer_finger",
        "right_inner_finger",
        "right_inner_knuckle",
        "right_outer_knuckle",
    ],
    "left_outer_finger": [
        "left_inner_finger",
        "left_inner_knuckle",
        "left_outer_knuckle",
        "right_outer_finger",
        "right_inner_finger",
        "right_inner_knuckle",
        "right_outer_knuckle",
    ],
    "left_inner_finger": [
        "left_inner_knuckle",
        "left_outer_knuckle",
        "right_outer_finger",
        "right_inner_finger",
        "right_inner_knuckle",
        "right_outer_knuckle",
    ],
    "left_inner_knuckle": [
        "left_outer_knuckle",
        "right_outer_finger",
        "right_inner_finger",
        "right_inner_knuckle",
        "right_outer_knuckle",
    ],
    "left_outer_knuckle": [
        "right_outer_finger",
        "right_inner_finger",
        "right_inner_knuckle",
        "right_outer_knuckle",
    ],
    "right_outer_finger": [
        "right_inner_finger",
        "right_inner_knuckle",
        "right_outer_knuckle",
    ],
    "right_inner_finger": ["right_inner_knuckle", "right_outer_knuckle"],
    "right_inner_knuckle": ["right_outer_knuckle"],
    "right_outer_knuckle": [],
    "right_arm_link2": ["right_arm_link3", "right_arm_link4", "right_arm_link5"],
    "right_arm_link3": ["right_arm_link4", "right_arm_link5"],
    "right_arm_link4": ["right_arm_link5", "right_arm_link6"],
    "right_arm_link5": ["right_arm_link6", "right_arm_link7"],
    "right_arm_link6": ["right_arm_link7", "right_robotiq_arg2f_base_link"],
    "right_arm_link7": ["right_robotiq_arg2f_base_link"],
    "right_robotiq_arg2f_base_link": [
        "right_robotiq_left_outer_finger",
        "right_robotiq_left_inner_finger",
        "right_robotiq_left_inner_knuckle",
        "right_robotiq_left_outer_knuckle",
        "right_robotiq_right_outer_finger",
        "right_robotiq_right_inner_finger",
        "right_robotiq_right_inner_knuckle",
        "right_robotiq_right_outer_knuckle",
    ],
    "right_robotiq_left_outer_finger": [
        "right_robotiq_left_inner_finger",
        "right_robotiq_left_inner_knuckle",
        "right_robotiq_left_outer_knuckle",
        "right_robotiq_right_outer_finger",
        "right_robotiq_right_inner_finger",
        "right_robotiq_right_inner_knuckle",
        "right_robotiq_right_outer_knuckle",
    ],
    "right_robotiq_left_inner_finger": [
        "right_robotiq_left_inner_knuckle",
        "right_robotiq_left_outer_knuckle",
        "right_robotiq_right_outer_finger",
        "right_robotiq_right_inner_finger",
        "right_robotiq_right_inner_knuckle",
        "right_robotiq_right_outer_knuckle",
    ],
    "right_robotiq_left_inner_knuckle": [
        "right_robotiq_left_outer_knuckle",
        "right_robotiq_right_outer_finger",
        "right_robotiq_right_inner_finger",
        "right_robotiq_right_inner_knuckle",
        "right_robotiq_right_outer_knuckle",
    ],
    "right_robotiq_left_outer_knuckle": [
        "right_robotiq_right_outer_finger",
        "right_robotiq_right_inner_finger",
        "right_robotiq_right_inner_knuckle",
        "right_robotiq_right_outer_knuckle",
    ],
    "right_robotiq_right_outer_finger": [
        "right_robotiq_right_inner_finger",
        "right_robotiq_right_inner_knuckle",
        "right_robotiq_right_outer_knuckle",
    ],
    "right_robotiq_right_inner_finger": [
        "right_robotiq_right_inner_knuckle",
        "right_robotiq_right_outer_knuckle",
    ],
    "right_robotiq_right_inner_knuckle": ["right_robotiq_right_outer_knuckle"],
    "right_robotiq_right_outer_knuckle": [],
}

GALBOT_INIT_QPOS = [
    -0.4,
    -1.2,
    -0.6,
    -1.7,
    0.3,
    0.4,
    -0.6,
] + [0.0 for _ in range(6)]

GALBOT_INIT_QPOS = [
    -0.92466492,
    -1.20374452,
    -0.36648354,
    -1.77962893,
    -1.36637611,
    -0.32985586,
    -0.86126989,
] + [0.0 for _ in range(6)]

GALBOT_INIT_QPOS = [
    -1.00481327,
    -1.18772186,
    -0.39541832,
    -1.74358496,
    -1.34655425,
    -0.35969566,
    -0.98926903,
] + [0.0 for _ in range(6)]

ROBOTIQ_WIDTH_MAX = 0.088
ROBOTIQ_GRIPPER_LONG = np.array([0, 0.0315, -0.0041]) + np.array([0, 0.0061, 0.0471])
ROBOTIQ_GRIPPER_THETA = np.arctan2(ROBOTIQ_GRIPPER_LONG[2], ROBOTIQ_GRIPPER_LONG[1])
ROBOTIQ_GRIPPER_LENGTH = np.linalg.norm(ROBOTIQ_GRIPPER_LONG[1:])


def COMPUTE_ROBOTIQ_GRIPPER_LONG(theta: float):
    theta2 = ROBOTIQ_GRIPPER_THETA + theta
    return np.array([0, np.cos(theta2), np.sin(theta2)]) * ROBOTIQ_GRIPPER_LENGTH


def COMPUTE_ROBOTIQ_GRIPPER_TIP(theta: float):
    return (
        np.array([0, 0.0306011, 0.054904])
        + COMPUTE_ROBOTIQ_GRIPPER_LONG(theta)
        + np.array([0.0, -0.024, 0.084])
    )


def COMPUTE_ROBOTIQ_DELTA_TCP(theta: float):
    return np.array([COMPUTE_ROBOTIQ_GRIPPER_TIP(theta)[2], 0, 0]) - np.array(
        [COMPUTE_ROBOTIQ_GRIPPER_TIP(0.0)[2], 0, 0]
    )


def COMPUTE_ROBOTIQ_GRIPPER_WIDTH(theta: float):
    return COMPUTE_ROBOTIQ_GRIPPER_TIP(theta)[1] * 2


def COMPUTE_ROBOTIQ_GRIPPER_ANGLE(width: float):
    width = np.clip(width, 0.0, ROBOTIQ_WIDTH_MAX)
    gripper_long_y = width / 2 - 0.0306011 + 0.024
    cos_theta2 = gripper_long_y / ROBOTIQ_GRIPPER_LENGTH
    theta2 = np.arccos(cos_theta2)
    theta = theta2 - ROBOTIQ_GRIPPER_THETA
    return theta


def GALBOT_GRIPPER_WIDTH_TO_ANGLE_DEPTH(width: float) -> Tuple[float, float]:
    angle = COMPUTE_ROBOTIQ_GRIPPER_ANGLE(width)
    depth = COMPUTE_ROBOTIQ_DELTA_TCP(angle)[0]
    return angle, depth


GalbotCfg = RobotCfg(
    path_urdf=os.path.join("asset", "robot", "galbot", "galbot_left_arm_gripper.urdf"),
    path_mjcf=os.path.join("asset", "robot", "galbot", "galbot_left_arm_gripper.xml"),
    path_collision=os.path.join("asset", "robot", "galbot", "galbot_robotiq.yml"),
    link_eef="left_gripper_tcp_link",
    joint_names=LEFT_ARM_JOINTS + ROBOTIQ_JOINTS,
    joint_init_qpos=np.array(GALBOT_INIT_QPOS),
    joint_arm_indices=list(range(7)),
    joint_gripper_indices=list(range(7, 13)),
    collision_ignore=GALBOT_CHARLIE_COLLISION_IGNORE,
    camera_cfg=WRIST_CAMERA,
    gripper_width_to_angle_depth=GALBOT_GRIPPER_WIDTH_TO_ANGLE_DEPTH,
)


def get_robot_cfg(robot: str) -> RobotCfg:
    if robot == "galbot":
        return deepcopy(GalbotCfg)
