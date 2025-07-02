import os
from contextlib import contextmanager
from pathlib import Path
import time
from typing import Union
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

import mujoco
import mujoco.viewer
from mujoco.viewer import Handle
from dm_control import mjcf
from dm_control.mujoco import MovableCamera

from src.type import Obj, Box, Mesh
from src.sim.cfg import MjSimConfig
from .constants import (
    DEFAULT_MJSCENE,
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


class MjSim:
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    viewer: Handle
    mjcf_root: mjcf.RootElement
    physics: mjcf.Physics
    renderer: MovableCamera

    def __init__(self, sim_cfg: MjSimConfig):
        self.cfg = sim_cfg
        self.robot_cfg = sim_cfg.robot_cfg
        self.headless = sim_cfg.headless
        self.ctrl_dt = sim_cfg.ctrl_dt
        self.sim_dt = sim_cfg.sim_dt
        self.viewer_cfg = sim_cfg.viewer_cfg
        self.render_cfg = sim_cfg.renderer_cfg
        self.use_debug_robot = sim_cfg.use_debug_robot

        self.mjcf_root = mjcf.from_xml_string(DEFAULT_MJSCENE)
        if self.cfg.use_ground_plane:
            self.mjcf_root.worldbody.add("geom", **DEFAULT_GROUD_GEOM)
        self._load_robot(
            self.cfg.robot_cfg.path_mjcf, add_debug_robot=self.use_debug_robot
        )
        # buffers
        self.launched = False

        self._body_name_ids = {}
        self.robot_joint_ids = []
        self.robot_actuator_ids = []
        self.debug_robot_joint_ids = []
        self.debug_actuator_ids = []
        self.geom_dict = {}

    def launch(self):
        # launch simulator
        self.physics = mjcf.Physics.from_mjcf_model(self.mjcf_root)
        self.mj_model = self.physics.model._model
        self.mj_data = self.physics.data._data

        # statistics
        debug_joint_names = [f"debug_{name}" for name in self.robot_cfg.joint_names]
        for joint_id, joint in enumerate(self.mjcf_root.find_all("joint")):
            if joint.name in self.robot_cfg.joint_names:
                self.robot_joint_ids.append(joint_id)
            if self.use_debug_robot and joint.name in debug_joint_names:
                self.debug_robot_joint_ids.append(joint_id)
        self.debug_qpos = np.zeros(len(self.debug_robot_joint_ids))

        for actuator_id, actuator in enumerate(self.mjcf_root.find_all("actuator")):
            if "debug" not in actuator.name:
                self.robot_actuator_ids.append(actuator_id)
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

        if self.render_cfg is not None:
            self.renderer = MovableCamera(
                self.physics, height=self.render_cfg.height, width=self.render_cfg.width
            )
            self.renderer.set_pose(
                self.render_cfg.lookat,
                self.render_cfg.distance,
                np.rad2deg(self.render_cfg.azimuth),
                np.rad2deg(self.render_cfg.elevation),
            )
        self.launched = True

    def reset(self, init_qpos):
        assert self.launched, "Simulator not launched"
        self.mj_data.qpos[self.robot_joint_ids] = init_qpos
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0
        if self.use_debug_robot:
            self.mj_data.qpos[self.debug_robot_joint_ids] = self.debug_qpos

        mujoco.mj_forward(self.mj_model, self.mj_data)
        if not self.headless:
            self.viewer.sync()

    def step(self, action):
        assert self.launched, "Simulator not launched"

        self.mj_data.ctrl[self.robot_actuator_ids] = action
        for _ in range(int(self.ctrl_dt / self.sim_dt)):
            mujoco.mj_step(self.mj_model, self.mj_data)
            if self.cfg.realtime_sync:
                time.sleep(self.sim_dt)

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

    def render(self, render_cfg=None):
        mode_kwargs = dict(
            rgb=dict(),
            depth=dict(depth=True),
            seg=dict(segmentation=True),
        )

        if render_cfg is not None:
            renderer = MovableCamera(
                self.physics, height=render_cfg.height, width=render_cfg.width
            )
            if render_cfg.fovy is not None:
                renderer._physics.model.vis.global_.fovy = render_cfg.fovy
            renderer.set_pose(
                render_cfg.lookat,
                render_cfg.distance,
                np.rad2deg(render_cfg.azimuth),
                np.rad2deg(render_cfg.elevation),
            )
        else:
            renderer = self.renderer
        return {k: renderer.render(**v).copy() for k, v in mode_kwargs.items()}

    def close(self):
        if not self.headless:
            self.viewer.close()
        self.physics.free()
        self.launched = False

    def _load_robot(self, robot_mjcf: str, add_debug_robot=False):
        assert robot_mjcf is not None, f"robot_mjcf is None, {robot_mjcf}"
        robot_mjcf = os.path.abspath(robot_mjcf)

        with temp_work_dir(robot_mjcf):
            robot_xml = mjcf.from_file(robot_mjcf)
            self.mjcf_root.attach(robot_xml)
            if add_debug_robot:
                debug_robot = robot_xml.__copy__()
                for body in debug_robot.find_all("body"):
                    if body.inertial is not None:
                        body.inertial.mass = mujoco.mjMINVAL

                for joint in debug_robot.find_all("joint"):
                    joint.name = f"debug_{joint.name}"
                    for k, v in joint.get_attributes().items():
                        if k not in ["dclass", "name", "pos", "axis", "range"]:
                            delattr(joint, k)
                        joint.damping = mujoco.mjMAXVAL
                del debug_robot.actuator

                for body in debug_robot.find_all("body"):
                    _remove_list = []
                    for geom_id in range(len(body.geom)):
                        if body.geom[geom_id].dclass is not None:
                            if body.geom[geom_id].dclass.dclass == "collision":
                                _remove_list.append(geom_id)
                            body.geom[geom_id].rgba = [1.0, 0.0, 0.0, 0.2]
                    _remove_list.reverse()
                    for geom_id in _remove_list:
                        del body.geom[geom_id]
                    body.name = f"debug_{body.name}"
                self.mjcf_root.attach(debug_robot)

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

    def get_seg_id_list(self, name: str):
        return [
            self.physics.model.name2id(geom_name, "geom")
            for geom_name in self.geom_dict[name]
        ]

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
    def robot_qpos(self):
        return self.mj_data.qpos[self.robot_joint_ids]

    def get_body_id(self, name):
        return self._body_name_ids.get(name, -1)

    def get_body_pose(self, body_name):
        body_id = self.get_body_id(body_name)
        pos = self.mj_data.xpos[body_id]
        rot = self.mj_data.xmat[body_id].reshape(3, 3)
        return pos, rot

    def debug_set_robot_qpos(self, qpos):
        if self.use_debug_robot:
            self.debug_qpos[:] = qpos
            self.mj_data.qpos[self.debug_robot_joint_ids] = qpos
            self.mj_data.qvel[self.debug_robot_joint_ids] = np.zeros_like(qpos)
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.viewer.sync()

    def debug_set_pose(self, pose):
        debug_body_id = self.mj_model.body(DEBUG_AXIS_BODY_NAME).id
        self.mj_data.mocap_pos[0] = pose[:3, 3]
        self.mj_data.mocap_quat[0] = mat2quat(pose[:3, :3])
        if not self.headless:
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.viewer.sync()
