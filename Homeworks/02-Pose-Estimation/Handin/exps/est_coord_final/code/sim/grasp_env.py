import os
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from src.type import Box, Mesh, Scene, Grasp
from src.utils import to_pose, rand_rot_mat, get_pc, get_workspace_mask
from src.constants import DEPTH_IMG_SCALE, TABLE_HEIGHT
from src.robot.cfg import get_robot_cfg
from src.robot.robot_model import RobotModel
from src.sim.mujoco import MjSim
from src.sim.cfg import MjSimConfig, MjRenderConfig
from src.vis import Vis


@dataclass
class Obs:
    rgb: np.ndarray
    """(H, W, 3) in RGB order"""
    depth: np.ndarray
    """(H, W) in meters"""
    seg: np.ndarray
    """(H, W) in uint8, 255 for object, 0 for background"""
    camera_pose: np.ndarray
    """(4, 4) camera pose in world frame"""
    object_pose: np.ndarray
    """(4, 4) object pose in world frame"""
    camera_frame_pc: Optional[np.ndarray] = None
    """(N, 3) point cloud in camera frame"""
    robot_frame_pc: Optional[np.ndarray] = None
    """(N, 3) point cloud in robot frame"""


@dataclass
class GraspEnvConfig:
    robot: str
    obj_name: str
    headless: bool
    ctrl_dt: float = 0.1
    wait_steps: int = 50
    prepare_steps: int = 50
    reach_steps: int = 10
    squeeze_steps: int = 10
    lift_steps: int = 10
    delta_dist: float = 0.01
    succ_height_thresh: float = 0.05
    grasp_trans_z_thresh: float = 0.01
    grasp_trans_z_violate: float = 0.025
    obj_pose: Optional[np.ndarray] = None


class GraspEnv:
    def __init__(self, config: GraspEnvConfig):
        """Initialize the grasp environment."""
        self.config = config
        self.robot_cfg = get_robot_cfg(config.robot)
        self.robot_model = RobotModel(self.robot_cfg)
        self.obj_name = config.obj_name

    def launch(self):
        """launch the simulation."""
        config = self.config
        self.sim = MjSim(
            MjSimConfig(
                robot_cfg=self.robot_cfg,
                headless=config.headless,
                realtime_sync=not config.headless,
                use_ground_plane=False,
                ctrl_dt=config.ctrl_dt,
            )
        )
        scene = get_default_scene()
        if config.obj_pose is None:
            obj_init_trans = np.array([0.45, 0.2, 0.6])
            obj_init_trans[:2] += np.random.uniform(-0.05, 0.05, 2)
            obj_pose = to_pose(obj_init_trans, rand_rot_mat())
        else:
            obj_pose = config.obj_pose.copy()
            obj_pose[2, 3] += 0.0025
        self.obj = get_obj(self.obj_name, obj_pose)
        scene.obj_list.append(self.obj)
        for o in scene.obj_list:
            self.sim.add_obj(o)
        self.sim.launch()

    def reset(self):
        """reset the simulation so that the robot is in the initial position. we step the simulation for a few steps to make sure the environment is stable."""
        init_qpos = self.robot_cfg.joint_init_qpos.copy()
        self.sim.reset(init_qpos)
        _qpos = init_qpos[:8]
        for _ in range(self.config.wait_steps):
            self.sim.step(_qpos)

    def get_obs(self) -> Obs:
        """Get the observation from the simulation"""
        init_qpos = self.robot_cfg.joint_init_qpos.copy()
        cam_trans, cam_rot = self.robot_model.fk_camera(init_qpos)
        cam_pose = to_pose(cam_trans, cam_rot)
        render_cfg = MjRenderConfig.from_intrinsics_extrinsics(
            self.robot_cfg.camera_cfg.height,
            self.robot_cfg.camera_cfg.width,
            self.robot_cfg.camera_cfg.intrinsics,
            cam_pose.copy(),
        )
        x = self.sim.render(render_cfg)
        obj_ids = self.sim.get_seg_id_list(self.obj.name)
        seg = x["seg"][..., 0]
        obj_seg = np.zeros_like(seg).astype(np.uint8)
        for i in range(len(obj_ids)):
            obj_seg[seg == obj_ids[i]] = 255
        obj_pose = to_pose(*self.sim.get_body_pose(self.obj.geom_id))
        obs = Obs(
            rgb=x["rgb"],
            depth=x["depth"],
            seg=obj_seg,
            camera_pose=cam_pose,
            object_pose=obj_pose,
        )
        obs.camera_frame_pc, obs.robot_frame_pc = self.get_pc(obs)
        return obs

    def get_pc(self, obs: Obs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the point cloud in camera and robot frame."""
        intrinsics = self.robot_cfg.camera_cfg.intrinsics
        camera_frame_pc = get_pc(obs.depth, intrinsics)
        align_rot = np.diag([-1, -1, 1])
        robot_frame_pc = (
            np.einsum(
                "ab,bc,nc->na", obs.camera_pose[:3, :3], align_rot, camera_frame_pc
            )
            + obs.camera_pose[:3, 3]
        )
        return camera_frame_pc, robot_frame_pc

    def save_obs(self, obs: Obs, data_dir=None):
        """Save the observation to the specified directory."""
        if data_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            data_dir = os.path.join("data", "pose2", timestamp)
        os.makedirs(data_dir)
        Image.fromarray(obs.rgb).save(os.path.join(data_dir, "rgb.png"))
        Image.fromarray(
            (np.clip(obs.depth, 0, 2.0) * DEPTH_IMG_SCALE).astype(np.uint16)
        ).save(os.path.join(data_dir, "depth.png"))
        Image.fromarray(obs.seg).save(os.path.join(data_dir, "obj_seg.png"))
        np.save(os.path.join(data_dir, "camera_pose.npy"), obs.camera_pose)
        np.save(os.path.join(data_dir, "object_pose.npy"), obs.object_pose)

    def plan_grasp(self, grasp: Grasp, pc: np.ndarray) -> Optional[np.ndarray]:
        """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
        start_gripper_angle = 0.0
        _, depth = self.robot_cfg.gripper_width_to_angle_depth(grasp.width)
        grasp_trans = grasp.trans - depth * grasp.rot[:, 0]
        succ, grasp_arm_qpos = self.robot_model.ik(grasp_trans, grasp.rot)
        if not succ:
            return None

        def add_gripper_qpos(qpos, angle):
            return np.concatenate(
                [qpos, np.full(len(self.robot_cfg.joint_gripper_indices), angle)]
            )

        traj = [add_gripper_qpos(grasp_arm_qpos, start_gripper_angle)]
        pc_mask = get_workspace_mask(pc)
        if self.robot_model.check_collision(traj[0], pc[pc_mask])[0]:
            return None

        cur_trans, cur_rot, cur_qpos = (
            grasp_trans.copy(),
            grasp.rot.copy(),
            grasp_arm_qpos.copy(),
        )
        for _ in range(self.config.reach_steps):
            cur_trans = cur_trans - self.config.delta_dist * cur_rot[:, 0]
            succ, cur_qpos = self.robot_model.ik(
                cur_trans, cur_rot, cur_qpos, delta_thresh=0.5
            )
            if not succ:
                return None
            traj = [add_gripper_qpos(cur_qpos, start_gripper_angle)] + traj

        target_gripper_angle, _ = self.robot_cfg.gripper_width_to_angle_depth(0.0)

        for i in range(self.config.squeeze_steps):
            traj.append(
                add_gripper_qpos(
                    grasp_arm_qpos,
                    start_gripper_angle
                    + (target_gripper_angle - start_gripper_angle)
                    * (i + 1)
                    / self.config.squeeze_steps,
                )
            )

        cur_trans, cur_rot, cur_qpos = (
            grasp_trans.copy(),
            grasp.rot.copy(),
            grasp_arm_qpos.copy(),
        )
        for _ in range(self.config.lift_steps):
            cur_trans[2] += self.config.delta_dist
            succ, cur_qpos = self.robot_model.ik(
                cur_trans, cur_rot, cur_qpos, delta_thresh=0.5
            )
            if not succ:
                return None
            traj.append(add_gripper_qpos(cur_qpos, target_gripper_angle))

        return traj

    def execute_plan(self, traj: np.ndarray) -> bool:
        """Execute the planned trajectory in the simulation and check if the grasp was successful."""
        obj_init_z = self.sim.get_body_pose(self.obj.geom_id)[0][2]
        self.sim.reset(traj[0])
        for _ in range(3):
            self.sim.step(traj[0][:8])
        for qpos in traj:
            self.sim.step(qpos[:8])

        obj_final_z = self.sim.get_body_pose(self.obj.geom_id)[0][2]
        if obj_final_z - obj_init_z > self.config.succ_height_thresh:
            return True
        else:
            return False

    def close(self):
        """close the simulation."""
        self.sim.close()


def get_default_scene() -> Scene:
    ground = Box(
        name="ground",
        pose=to_pose(trans=np.array([0, 0, -0.05])),
        size=np.array([2, 2, 0.1]),
        fixed_body=True,
    )
    table = Box(
        name="table",
        pose=to_pose(trans=np.array([0.8, 0, 0.45])),
        size=np.array([1.2, 2.0, 0.1]),
        fixed_body=True,
    )

    scene = Scene(obj_list=[ground, table])
    # workspace = Box(
    #     name="workspace",
    #     pose=to_pose(trans=np.array([0.45, 0.2, 0.52])),
    #     size=np.array([0.3, 0.3, 0.01]),
    #     fixed_body=True,
    # )
    # scene.obj_list.append(workspace)
    return scene


def get_obj(name: str, pose: np.ndarray) -> Mesh:
    mesh_dir = os.path.join("asset", "obj", name)
    obj = Mesh(
        name=name,
        pose=pose,
        path=os.path.join(mesh_dir, "single.obj"),
        convex_decompose_paths=[
            os.path.join(mesh_dir, "decompose", x)
            for x in os.listdir(os.path.join(mesh_dir, "decompose"))
            if x.endswith(".obj")
        ],
    )
    return obj


def get_grasps(name: str) -> List[Grasp]:
    def flip_grasp_rot(g: Grasp):
        flip_rot = g.rot.copy()
        flip_rot[:, 1:] *= -1
        return Grasp(
            trans=g.trans.copy(),
            rot=flip_rot,
            width=g.width,
        )

    if name == "power_drill":
        orig_grasps = [
            Grasp(
                trans=np.array([-0.005, 0, 0.005]),
                rot=np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
                width=0.08,
            ),
            Grasp(
                trans=np.array([-0.005, 0, 0.025]),
                rot=np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
                width=0.08,
            ),
            Grasp(
                trans=np.array([-0.005, 0, 0.015]),
                rot=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                width=0.08,
            ),
            Grasp(
                trans=np.array([-0.005, 0, 0.015]),
                rot=np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                width=0.08,
            ),
            Grasp(
                trans=np.array([-0.005, 0.05, 0.016]),
                rot=np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                width=0.08,
            ),
            Grasp(
                trans=np.array([-0.005, -0.03, 0.016]),
                rot=np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
                width=0.08,
            ),
            Grasp(
                trans=np.array([0.016, 0.056, 0.015]),
                rot=np.array(
                    [
                        [-np.sqrt(2) / 2, 0, -np.sqrt(2) / 2],
                        [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
                        [0, 1, 0],
                    ]
                ),
                width=0.08,
            ),
            Grasp(
                trans=np.array([-0.02, 0.036, 0.015]),
                rot=np.array(
                    [
                        [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
                        [np.sqrt(2) / 2, 0, -np.sqrt(2) / 2],
                        [0, 1, 0],
                    ]
                ),
                width=0.08,
            ),
        ]

    ret = []
    for g in orig_grasps:
        ret.append(g)
        # we can flip the grasp around the gripper's forward axis since the gripper is symmetric
        ret.append(flip_grasp_rot(g))
    return ret
