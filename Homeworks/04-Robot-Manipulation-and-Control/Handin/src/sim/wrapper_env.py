import os
import time
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np
from transforms3d.quaternions import quat2mat
from PIL import Image

from src.type import Box, Mesh, Scene, Grasp
from src.utils import to_pose, rand_rot_mat
from src.constants import DEPTH_IMG_SCALE
from src.robot.cfg import get_robot_cfg, get_quad_cfg
from src.robot.robot_model import RobotModel
from src.sim.combined_sim import CombinedSim
from src.sim.cfg import CombinedSimConfig, MjRenderConfig
import onnxruntime as rt
from src.quad_highlevel.deploy.sim.quad_sim import PlayGo2LocoEnv
import src.quad_highlevel.deploy.constants as quad_consts


@dataclass
class Obs:
    rgb: np.ndarray
    """(H, W, 3) in RGB order"""
    depth: np.ndarray
    """(H, W) in meters"""
    camera_pose: np.ndarray
    """(4, 4) camera pose in world frame"""

@dataclass
class WrapperEnvConfig:
    humanoid_robot: str
    obj_name: str
    headless: bool
    ctrl_dt: float = 0.02
    reset_wait_steps: int = 100
    quad_load_pos: np.ndarray = np.array([2.0, 1.0, 0.45])
    quad_reset_pos: np.ndarray = np.array([2.0, -0.2, 0.278])
    obj_pose: Optional[np.ndarray] = None
    table_pose: Optional[np.ndarray] = None
    table_size: Optional[np.ndarray] = None

class WrapperEnv:
    def __init__(self, config: WrapperEnvConfig):
        """Initialize the grasp environment."""
        self.config = config
        self.humanoid_robot_cfg = get_robot_cfg(config.humanoid_robot)
        self.humanoid_robot_model = RobotModel(self.humanoid_robot_cfg)
        self.quad_robot_cfg = get_quad_cfg()
        self.obj_name = config.obj_name


    def set_table_obj_config(self, table_pose: np.ndarray, table_size: np.ndarray, obj_pose: np.ndarray):
        """Set the table pose and size and object pose."""
        self.config.table_pose = table_pose
        self.config.table_size = table_size
        self.config.obj_pose = obj_pose
    
    def set_quad_reset_pos(self, quad_reset_pos: np.ndarray):
        """Set the quad load position."""
        self.config.quad_reset_pos = quad_reset_pos

    def launch(self):
        """launch the simulation."""
        config = self.config
        self.sim = CombinedSim(
            CombinedSimConfig(
                humanoid_robot_cfg=self.humanoid_robot_cfg,
                quad_robot_cfg=self.quad_robot_cfg,
                headless=config.headless,
                realtime_sync=not config.headless,
                use_ground_plane=False,
                ctrl_dt=config.ctrl_dt,
                quad_load_pos=config.quad_load_pos,
                quad_reset_pos=config.quad_reset_pos,
            )
        )
        self.quad_sim = PlayGo2LocoEnv(headless=True)
        self.quad_info, self.quad_obs = None, None

        self.quad_output_names = ["continuous_actions"]
        self.quad_policy = rt.InferenceSession(quad_consts.POLICY_ONNX_PATH, providers=["CPUExecutionProvider"])


        if self.config.table_pose is not None and self.config.table_size is not None:
            table_pose = self.config.table_pose
            table_size = self.config.table_size
        else:
            self.config.table_pose=to_pose(trans=np.array([0.6, 0.35, 0.72]))
            self.config.table_size=np.array([0.68, 0.36, 0.02])
            table_pose = self.config.table_pose
            table_size = self.config.table_size
        scene = get_scene_table(table_pose, table_size)

        if self.config.obj_pose is not None:
            obj_pose = self.config.obj_pose
        else:
            obj_init_trans = np.array([0.5, 0.3, 0.82])
            obj_init_trans[:2] += np.random.uniform(-0.02, 0.02, 2) * 0
            obj_pose = to_pose(obj_init_trans, rand_rot_mat())
        
        self.obj = get_obj(self.obj_name, obj_pose)
        scene.obj_list.append(self.obj)
        for o in scene.obj_list:
            self.sim.add_obj(o)
        self.sim.launch()

    def reset(self, humanoid_head_qpos: Optional[np.ndarray] = None, humanoid_qpos: Optional[np.ndarray] = None, reset_wait_steps: Optional[int] = None):
        """reset the simulation so that the robot is in the initial position. we step the simulation for a few steps to make sure the environment is stable."""
        if humanoid_qpos is None:
            humanoid_init_qpos = self.humanoid_robot_cfg.joint_init_qpos.copy()
        else:
            humanoid_init_qpos = humanoid_qpos
        if humanoid_head_qpos is None:
            humanoid_head_qpos = np.array([0.0, 0.0])

        self.sim._fix_driller = False
        self.sim._fix_pose_quad_to_driller = None
        self.sim._driller_valid_count = 0

        self.sim.reset(humanoid_head_qpos, humanoid_init_qpos, self.quad_robot_cfg.joint_init_qpos)
        
        self.quad_info, self.quad_obs = self.quad_sim.reset(
            base_pose=self.sim.default_quad_pose,
            qpos=self.sim.quad_robot_cfg.joint_init_qpos,
        )
        
        if reset_wait_steps is None:
            reset_wait_steps = self.config.reset_wait_steps
        
        for _ in range(reset_wait_steps):
            self.sim.step_reset()
        
        self._init_container_pose = self.get_container_pose()

    def get_obs(self, camera_id: int = 1) -> Obs:
        """Get the observation from the simulation, camera_id = 0 for head camera, camera_id = 1 for wrist camera."""
        qpos = self.get_state()
        if camera_id == 1:
            cam_trans, cam_rot = self.humanoid_robot_model.fk_camera(qpos, camera_id)
        elif camera_id == 0:
            cam_trans, cam_rot = self.humanoid_robot_model.fk_camera(self.sim.humanoid_head_qpos, camera_id)
        else:
            raise NotImplementedError
        
        cam_pose = to_pose(cam_trans, cam_rot)
        # render_cfg = MjRenderConfig.from_intrinsics_extrinsics(
        #     self.humanoid_robot_cfg.camera_cfg[camera_id].height,
        #     self.humanoid_robot_cfg.camera_cfg[camera_id].width,
        #     self.humanoid_robot_cfg.camera_cfg[camera_id].intrinsics,
        #     cam_pose.copy(),
        # )
        x = self.sim.render(camera_id=camera_id, camera_pose=cam_pose)

        obs = Obs(
            rgb=x["rgb"],
            depth=x["depth"],
            camera_pose=cam_pose,
        )

        return obs
    
    def get_state(self) -> Dict:
        humanoid_qpos = self.sim.mj_data.qpos[self.sim.humanoid_joint_ids]
        return humanoid_qpos

    def step_env(
        self,
        humanoid_head_qpos: Optional[np.ndarray] = None,
        humanoid_action: Optional[np.ndarray] = None,
        quad_command: Optional[np.ndarray] = None,
        gripper_open: Optional[int] = None,
    ):
        """
        Step the simulation with the given humanoid head qpos and action and quad robot command.
        humanoid_head_qpos: (2,)
        humanoid_action: (7,) 7 dof freedom
        quad_command: (3,) quadrotor command (v_x, v_y, omega_z)
        gripper_open: 1 for open, 0 for close
        """
        assert not (humanoid_action is not None and gripper_open is not None), "Cannot specify both humanoid_action and gripper_open at the same time."
        
        if quad_command is None:
            quad_command = np.array([0.0, 0.0, 0.0])
        else:
            assert len(quad_command) == 3
        
        self.quad_info["command"] = quad_command.copy()
        onnx_input = {"obs": self.quad_obs["state"].reshape(1, -1).astype(np.float32)}
        quad_action = self.quad_policy.run(self.quad_output_names, onnx_input)[0][0]
        self.quad_info, self.quad_obs , quad_poses, quad_qposes = self.quad_sim.step(
            self.quad_info, quad_action, return_all=True
        )
        if humanoid_action is None:
            if gripper_open is not None:
                gripper_angle = 0 if gripper_open else 0.83432372
                humanoid_action = self.sim.cached_humanoid_action[:7].copy()
                full_humanoid_action = np.concatenate(
                    [humanoid_action, np.array([gripper_angle])]
                )
            else:
                full_humanoid_action = None
        else:
            assert len(humanoid_action) == 7
            full_humanoid_action = np.concatenate(
                [humanoid_action, np.array([self.sim.cached_humanoid_action[7]])]
            )

        self.sim.step(
            humanoid_head_qpos=humanoid_head_qpos,
            humanoid_arm_action=full_humanoid_action,
            quad_poses=quad_poses,
            quad_qposes=quad_qposes,
        )

        if not self.sim._fix_driller:
            # check if the driller is in the container
            if self.detect_drop_precision():
                self.sim._driller_valid_count += 1
                if self.sim._driller_valid_count > 8:
                    self.sim._fix_driller = True  
                    driller_pose = self.get_driller_pose().copy()
                    driller_pose[2, 3] += 0.005 # raise the driller a bit
                    quad_pose = self.get_quad_pose()
                    self.sim._fix_pose_quad_to_driller = np.linalg.inv(quad_pose) @ driller_pose
                    


    def debug_save_obs(self, obs: Obs, data_dir=None):
        """Save the observation to the specified directory."""
        if data_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            data_dir = os.path.join("data", "pose2", timestamp)
        os.makedirs(data_dir, exist_ok=True)
        # clear the directory
        for f in os.listdir(data_dir):
            os.remove(os.path.join(data_dir, f))
        Image.fromarray(obs.rgb).save(os.path.join(data_dir, "rgb.png"))
        Image.fromarray(
            (np.clip(obs.depth, 0, 2.0) * DEPTH_IMG_SCALE).astype(np.uint16)
        ).save(os.path.join(data_dir, "depth.png"))
        np.save(os.path.join(data_dir, "camera_pose.npy"), obs.camera_pose)

    def get_quad_pose(self) -> np.ndarray:
        pose_array = self.sim._debug_get_quad_pose()
        trans = pose_array[:3].copy()
        rot = quat2mat(pose_array[3:])
        return to_pose(trans, rot)
    
    def get_container_pose(self) -> np.ndarray:
        pose_array = self.sim._debug_get_quad_pose()
        trans = np.asanyarray(pose_array[:3].copy())
        rot = quat2mat(pose_array[3:])
        trans += rot @ np.array([-0.09, 0, 0.0455 + 0.05])
        return to_pose(trans, rot)

    def get_driller_pose(self) -> np.ndarray:
        pose_array = self.sim._debug_get_driller_pose()
        trans = pose_array[:3].copy()
        rot = quat2mat(pose_array[3:])
        return to_pose(trans, rot)

    def metric_obj_pose(self, obj_pose:np.ndarray):
        """
        judge the obj_pose is correct or not
        """
        driller_pose = self.get_driller_pose()
        dist_diff = np.linalg.norm(driller_pose[:3, 3] - obj_pose[:3, 3])
        rot_diff = driller_pose[:3, :3] @ obj_pose[:3, :3].T
        angle_diff = np.abs(np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)))
        if dist_diff < 0.025 and angle_diff < 0.25:
            return True
        return False
    
    def detect_drop_precision(self):
        """
        judge the drop precision
        """
        driller_pose = self.get_driller_pose()
        container_pose = self.get_container_pose()
        container_rot = container_pose[:3, :3].copy()
        driller_trans = driller_pose[:3, 3].copy()
        driller_trans_container = container_rot.T @ (driller_trans - container_pose[:3, 3])
        container_size = np.array([0.3, 0.2, 0.1])
        # check if the driller is in the container
        if np.abs(driller_trans_container[0]) < container_size[0] / 2 and \
            np.abs(driller_trans_container[1]) < container_size[1] / 2 and \
            np.abs(driller_trans_container[2]) < container_size[2] / 2:
            return True   
        return False
    
    def metric_drop_precision(self):
        return self.sim._fix_driller
    
    def metric_quad_return(self):
        """
        judge if the quadruped's box returned to the original position
        """
        container_pose = self.get_container_pose()
        container_trans = container_pose[:3, 3].copy()
        init_quad_trans = self._init_container_pose[:3, 3].copy()
        dist_xy = np.linalg.norm(container_trans[:2] - init_quad_trans[:2])

        if dist_xy < 0.1:
            return True
        return False
    
    def close(self):
        """close the simulation."""
        self.sim.close()

def get_scene_table(table_pose:np.ndarray, table_size:np.ndarray) -> Scene:
    table = Box(
        name="table",
        pose=table_pose,
        size=table_size,
        fixed_body=True,
    )
    pose_table_leg1 = table_pose.copy()
    pose_table_leg1[0,3] -= table_size[0] / 2 - 0.14
    pose_table_leg1[2,3] /= 2
    size_table_leg = table_size.copy()
    size_table_leg[0] = 0.05
    size_table_leg[1] = 0.05
    size_table_leg[2] = table_pose[2,3]
    pose_table_leg2 = pose_table_leg1.copy()
    pose_table_leg2[0,3] += table_size[0] - 0.28

    table_leg1 = Box(
        name="table_leg1",
        pose=pose_table_leg1,
        size=size_table_leg,
        fixed_body=True,
    )    
    # table_leg2 = Box(
    #     name="table_leg2",
    #     pose=pose_table_leg2,
    #     size=size_table_leg,
    #     fixed_body=True,
    # )
    pose_table_bottom1 = table_pose.copy()
    pose_table_bottom1[0,3] = pose_table_leg1[0,3] - 0.07
    pose_table_bottom1[2,3] = 0.01
    size_table_bottom = table_size.copy()
    size_table_bottom[0] = 0.08
    size_table_bottom[2] = 0.02
    pose_table_bottom2 = pose_table_bottom1.copy()
    pose_table_bottom2[0,3] = pose_table_leg2[0,3] + 0.07
    pose_table_bottom3 = table_pose.copy()
    pose_table_bottom3[2,3] = 0.01
    size_table_bottom3 = table_size.copy()
    size_table_bottom3[0] = pose_table_bottom2[0,3] - pose_table_bottom1[0,3] - 0.08
    size_table_bottom3[1] = 0.08
    size_table_bottom3[2] = 0.02 

    table_bottom1 = Box(
        name="table_bottom1",
        pose=pose_table_bottom1,
        size=size_table_bottom,
        fixed_body=True,
    )
    table_bottom2 = Box(
        name="table_bottom2",
        pose=pose_table_bottom2,
        size=size_table_bottom,
        fixed_body=True,
    )
    table_bottom3 = Box(
        name="table_bottom3",
        pose=pose_table_bottom3,
        size=size_table_bottom3,
        fixed_body=True,
    )
    
    scene = Scene(obj_list=[table, table_leg1, table_bottom1, table_bottom2, table_bottom3])
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

