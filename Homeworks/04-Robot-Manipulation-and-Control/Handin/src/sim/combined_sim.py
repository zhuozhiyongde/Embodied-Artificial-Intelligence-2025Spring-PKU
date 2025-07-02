import os
from contextlib import contextmanager
from pathlib import Path
import time
from typing import Union, List
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

import mujoco
import mujoco.viewer
from mujoco.viewer import Handle
from dm_control import mjcf
from dm_control.mujoco import MovableCamera

from src.utils import to_pose
from src.type import Obj, Box, Mesh
from src.sim.cfg import CombinedSimConfig
from .constants import (
    DEFAULT_COMBINED_MJSCENE,
    DEFAULT_GROUD_GEOM,
    DEBUG_AXIS_BODY_NAME,
)


@contextmanager
def temp_work_dir(target_path):
    """Temporarily change the current working directory."""
    original_dir = os.getcwd()
    if Path(target_path).is_file():
        target_dir = Path(target_path).parent
    else:
        target_dir = target_path

    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(original_dir)


class CombinedSim:
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    viewer: Handle
    mjcf_root: mjcf.RootElement
    physics: mjcf.Physics

    def __init__(self, sim_cfg: CombinedSimConfig):
        self.cfg = sim_cfg
        self.humanoid_robot_cfg = sim_cfg.humanoid_robot_cfg
        # self.quad_robot_path_mjcf = sim_cfg.quad_robot_path_mjcf
        self.quad_robot_cfg = sim_cfg.quad_robot_cfg
        self.quad_action_scale = sim_cfg.quad_action_scale
        self.headless = sim_cfg.headless
        self.ctrl_dt = sim_cfg.ctrl_dt
        self.sim_dt = sim_cfg.sim_dt
        self.viewer_cfg = sim_cfg.viewer_cfg
        self.render_cfg = sim_cfg.renderer_cfg

        self.mjcf_root = mjcf.from_xml_string(DEFAULT_COMBINED_MJSCENE)
        if self.cfg.use_ground_plane:
            self.mjcf_root.worldbody.add("geom", **DEFAULT_GROUD_GEOM)
        self._load_robot_quad(self.quad_robot_cfg.path_mjcf, pos = sim_cfg.quad_load_pos)
        self._load_robot_humanoid(
            self.cfg.humanoid_robot_cfg.path_mjcf
        )
        # buffers
        self.launched = False

        self._body_name_ids = {}
        self.humanoid_joint_ids = []
        self.quad_joint_ids = []
        self.humanoid_actuator_ids = []
        self.quad_actuator_ids = []
        self.debug_robot_joint_ids = []
        self.debug_actuator_ids = []
        self.geom_dict = {}
        self.lock_step = False
        self._fix_driller = False
        self._fix_pose_quad_to_driller = None
        self._driller_valid_count = 0
        self.humanoid_head_qpos = np.zeros(2)

    def launch(self):
        self.physics = mjcf.Physics.from_mjcf_model(self.mjcf_root)
        self.mj_model = self.physics.model._model
        self.mj_data = self.physics.data._data

        head_camera_body_id = self.physics.model.name2id("camera_body_head", "body")
        wrist_camera_body_id = self.physics.model.name2id("camera_body_wrist", "body")
        head_camera_mocap_id = self.physics.model.body_mocapid[head_camera_body_id]
        wrist_camera_mocap_id = self.physics.model.body_mocapid[wrist_camera_body_id]
        head_camera_id = self.physics.model.name2id("moving_cam_head", "camera")
        wrist_camera_id = self.physics.model.name2id("moving_cam_wrist", "camera")
        head_camera_joint_name = "head_camera_freejoint"
        wrist_camera_joint_name = "wrist_camera_freejoint"

        height_head_camera = self.humanoid_robot_cfg.camera_cfg[0].height
        width_head_camera = self.humanoid_robot_cfg.camera_cfg[0].width
        height_wrist_camera = self.humanoid_robot_cfg.camera_cfg[1].height
        width_wrist_camera = self.humanoid_robot_cfg.camera_cfg[1].width

        fovy_head_camera = (
            2 * np.arctan(0.5 * height_head_camera / self.humanoid_robot_cfg.camera_cfg[0].intrinsics[1, 1])\
                  * (180 / np.pi)
        )  # in degrees
        fovy_wrist_camera = (
            2 * np.arctan(0.5 * height_wrist_camera / self.humanoid_robot_cfg.camera_cfg[1].intrinsics[1, 1])\
                  * (180 / np.pi)
        )
        self.physics.model.cam_fovy[head_camera_id] = fovy_head_camera
        self.physics.model.cam_fovy[wrist_camera_id] = fovy_wrist_camera

        self.render_height = [height_head_camera, height_wrist_camera]
        self.render_width = [width_head_camera, width_wrist_camera]

        self.camera_body_ids = [head_camera_body_id, wrist_camera_body_id]
        self.camera_ids = [head_camera_id, wrist_camera_id]
        self.camera_joint_names = [head_camera_joint_name, wrist_camera_joint_name]
        self.camera_mocap_ids = [head_camera_mocap_id, wrist_camera_mocap_id]
        # ctrl dof: 12+8
        # qpos dof: 19+20
        self.ctrl_humanoid_begin = 12
        self.qpos_humanoid_begin = 19

        # statistics
        for joint_id, joint in enumerate(self.mjcf_root.find_all("joint")):
            if joint.name in self.humanoid_robot_cfg.joint_names:
                self.humanoid_joint_ids.append(joint_id+6)

        for actuator_id, actuator in enumerate(self.mjcf_root.find_all("actuator")):
            if not 'left' in actuator.name:
                self.quad_actuator_ids.append(actuator_id)
                continue
            if "debug" not in actuator.name:
                self.humanoid_actuator_ids.append(actuator_id)
            else:
                self.debug_actuator_ids.append(actuator_id)

        for i in range(self.mj_model.nbody):
            self._body_name_ids[self.mj_model.body(i).name] = i

        # modules
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model,
                self.mj_data,
            )
            if self.viewer_cfg is not None:
                self.viewer.cam.lookat[:] = self.viewer_cfg.lookat
                self.viewer.cam.distance = self.viewer_cfg.distance
                self.viewer.cam.azimuth = np.rad2deg(self.viewer_cfg.azimuth)
                self.viewer.cam.elevation = np.rad2deg(self.viewer_cfg.elevation)

        # if self.render_cfg is not None:
        #     self.renderer = MovableCamera(
        #         self.physics, height=self.render_cfg.height, width=self.render_cfg.width
        #     )
        #     self.renderer.set_pose(
        #         self.render_cfg.lookat,
        #         self.render_cfg.distance,
        #         np.rad2deg(self.render_cfg.azimuth),
        #         np.rad2deg(self.render_cfg.elevation),
        #     )
        self.launched = True

    def reset(self, humanoid_head_qpos, humanoid_init_qpos, quad_init_joint_angles):
        assert self.launched, "Simulator not launched"
        self.humanoid_head_qpos = humanoid_head_qpos
        self.mj_data.qpos[self.humanoid_joint_ids] = humanoid_init_qpos.copy()
        # self.mj_data.qpos[:3] = self.cfg.quad_reset_pos
        self.mj_data.qpos[:7] = self.quad_robot_cfg.default_pose.copy()
        self.mj_data.qpos[:3] = self.cfg.quad_reset_pos.copy()
        self.mj_data.qpos[7:self.qpos_humanoid_begin] = quad_init_joint_angles.copy()
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.ctrl[self.quad_actuator_ids] = quad_init_joint_angles.copy()
        self.mj_data.ctrl[self.humanoid_actuator_ids] = humanoid_init_qpos[:8].copy()

        self.cached_humanoid_action = humanoid_init_qpos[:8].copy()

        self.default_quad_pose = self.mj_data.qpos[:7].copy()
        self.default_ctrl = self.mj_data.ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        if not self.headless:
            self.viewer.sync()
    
    def step_reset(self):
        self.mj_data.ctrl = self.default_ctrl.copy()
        for _ in range(int(self.ctrl_dt / self.sim_dt)):
            if not self.lock_step:
                mujoco.mj_step(self.mj_model, self.mj_data)
            if self.cfg.realtime_sync:
                time.sleep(self.sim_dt)

    def __set_driller_pose(self, driller_pose):
        self.physics.named.data.qpos['//unnamed_joint_1'] = driller_pose
        self.physics.forward()
    
    def __update_driller_pose(self):
        trans_quad = self.mj_data.qpos[:3]
        quat_quad = self.mj_data.qpos[3:7]
        rot_quad = quat2mat(quat_quad)
        pose_quad = to_pose(trans_quad, rot_quad)
        pose_driller = pose_quad @ self._fix_pose_quad_to_driller
        trans_driller = pose_driller[:3, 3]
        rot_driller = pose_driller[:3, :3]
        quat_driller = mat2quat(rot_driller)
        self.__set_driller_pose(np.concatenate([trans_driller, quat_driller]))

    def __get_driller_pose(self):
        return np.asanyarray(self.physics.named.data.qpos['//unnamed_joint_1']).copy()

    def step(self, humanoid_head_qpos, humanoid_arm_action, quad_poses, quad_qposes, update=1):
        if humanoid_head_qpos is not None: self.humanoid_head_qpos = humanoid_head_qpos
        if humanoid_arm_action is None: humanoid_arm_action = self.cached_humanoid_action.copy()
        else: self.cached_humanoid_action = humanoid_arm_action.copy()
        
        self.mj_data.ctrl[self.humanoid_actuator_ids] = humanoid_arm_action
        steps = int(self.ctrl_dt/self.sim_dt)
        assert steps==len(quad_poses)
        for i in range(steps):
            quad_pose = quad_poses[i]
            quad_qpos = quad_qposes[i]
            
            self.mj_data.qpos[:7] = quad_pose.copy()
            self.mj_data.qpos[7:self.qpos_humanoid_begin] = quad_qpos.copy()
            self.mj_data.qvel[:self.qpos_humanoid_begin-1] = 0.0
            self.mj_data.ctrl[self.quad_actuator_ids] = quad_qpos.copy()
            mujoco.mj_forward(self.mj_model, self.mj_data)
            for _ in range(update):
                mujoco.mj_step(self.mj_model, self.mj_data)
            if self.cfg.realtime_sync:
                time.sleep(self.sim_dt)
        if self._fix_driller:
            self.__update_driller_pose()
        if not self.headless:
            self.viewer.sync()

    def add_obj(self, obj: Obj):
        assert not self.launched, "Cannot add object after simulator is launched"
        assert isinstance(obj, Obj), f"{obj} must be instance of {Obj}"
        body = self.mjcf_root.worldbody.add(
            f"body",
            name=obj.geom_id,
            pos=obj.trans,
            quat=mat2quat(obj.rot),
        )
        if not obj.fixed_body:
            body.add("freejoint")

        if isinstance(obj, Box):
            self._add_geom_primitive(body, obj)
        elif isinstance(obj, Mesh):
            self._add_geom_meshes(body, obj)
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")

    def render(self, camera_id=0, camera_pose=None):
        mode_kwargs = dict(
            rgb=dict(),
            depth=dict(depth=True),
            seg=dict(segmentation=True),
        )
        if camera_pose is not None:
            if camera_id == 0:
                print(camera_pose)
            trans = camera_pose[:3, 3].copy()
            self.mj_data.mocap_pos[self.camera_mocap_ids[camera_id]] = trans
            quat = mat2quat(camera_pose[:3, :3].dot(np.diag([1,-1,-1])))
            self.mj_data.mocap_quat[self.camera_mocap_ids[camera_id]] = quat
            mujoco.mj_forward(self.mj_model, self.mj_data)
        # if render_cfg is not None:
        #     # self.physics.model.vis.global_.offwidth = render_cfg.width
        #     # self.physics.model.vis.global_.offheight = render_cfg.height
        #     renderer = MovableCamera(
        #         self.physics, height=render_cfg.height, width=render_cfg.width
        #     )
        #     if render_cfg.fovy is not None:
        #         renderer._physics.model.vis.global_.fovy = render_cfg.fovy
        #     renderer.set_pose(
        #         render_cfg.lookat,
        #         render_cfg.distance,
        #         np.rad2deg(render_cfg.azimuth),
        #         np.rad2deg(render_cfg.elevation),
        #     )
        # else:
        #     renderer = self.renderer
        # return {k: renderer.render(**v).copy() for k, v in mode_kwargs.items()}
        return {
            k: self.physics.render(
                height=self.render_height[camera_id],
                width=self.render_width[camera_id],
                camera_id=self.camera_ids[camera_id],
                **v
            ).copy() for k, v in mode_kwargs.items()
        }

    def close(self):
        if not self.headless:
            self.viewer.close()
        self.physics.free()
        self.launched = False

    def _load_robot_humanoid(self, robot_mjcf: str):
        assert robot_mjcf is not None, f"robot_mjcf is None, {robot_mjcf}"
        robot_mjcf = os.path.abspath(robot_mjcf)

        with temp_work_dir(robot_mjcf):
            robot_xml = mjcf.from_file(robot_mjcf)
            self.mjcf_root.attach(robot_xml)
    
    def _load_robot_quad(self, robot_mjcf: str, pos = [1,0,0.445]):
        robot_mjcf = os.path.abspath(robot_mjcf)
        # print(robot_mjcf)
        with temp_work_dir(robot_mjcf):
            quad_robot = mjcf.from_file(robot_mjcf)
            body = self.mjcf_root.attach(quad_robot)
            body.pos = pos
            body.add("freejoint")

    def _add_geom_primitive(self, body, obj: Box):
        # process obj prop
        geom_prop = dict(
            name=obj.name,
        )
        if obj.color is not None:
            geom_prop["rgba"] = (
                obj.color if len(obj.color) == 4 else np.append(obj.color, 1)
            )
        if isinstance(obj, Box):
            geom_prop["type"] = "box"
            geom_prop["size"] = np.float32(obj.size) / 2.0
        else:
            raise NotImplementedError
        body.add("geom", **geom_prop)
        self.geom_dict[obj.name] = [obj.name]

    def _add_geom_meshes(self, body, obj: Mesh):
        # mesh prop
        if obj.path is not None:
            if hasattr(obj, "convex_decompose_paths"):
                path_list_collision = obj.convex_decompose_paths
            else:
                path_list_collision = [obj.path]
            self.geom_dict[obj.name] = []
            for i, coll_path in enumerate(path_list_collision):
                geom_name = f"{obj.name}_collision_{i}"
                self.geom_dict[obj.name].append(geom_name)
                self._add_mesh(body, obj, geom_name, coll_path, False)

        else:
            raise NotImplementedError

    def _add_mesh(self, body, obj, mesh_name, path, is_visual=False):
        mesh_prop = dict(
            name=mesh_name,
            file=path,
        )
        if obj.scale is not None:
            assert len(obj.scale) == 3, f"scale must be 3D, {obj.scale}"
            mesh_prop["scale"] = obj.scale
        self.mjcf_root.asset.add("mesh", **mesh_prop)
        geom_prop = dict(
            name=mesh_name,
            type="mesh",
            mesh=mesh_name,
            condim=6,
        )
        body.add("geom", **geom_prop)

    @property
    def humanoid_robot_qpos(self):
        return self.mj_data.qpos[self.humanoid_joint_ids]

    def __get_body_id(self, name):
        return self._body_name_ids.get(name, -1)

    def __get_body_pose(self, body_name):
        body_id = self.__get_body_id(body_name)
        pos = self.mj_data.xpos[body_id]
        rot = self.mj_data.xmat[body_id].reshape(3, 3)
        return pos, rot
    
    def _debug_get_driller_pose(self):
        return self.__get_driller_pose()
    
    def _debug_get_body_pose(self, body_name):
        return self.__get_body_pose(body_name)
    
    def _debug_get_quad_pose(self):
        pose_array = self.mj_data.qpos[:7].copy()
        return pose_array
        
    def debug_vis_pose(self, pose):
        self.mj_data.mocap_pos[0] = pose[:3, 3]
        self.mj_data.mocap_quat[0] = mat2quat(pose[:3, :3])
        if not self.headless:
            mujoco.mj_forward(self.mj_model, self.mj_data)
            if self._fix_driller:
                self.__update_driller_pose()
            self.viewer.sync()
