
import time
import numpy as np

import mujoco
import mujoco.viewer

import src.quad_highlevel.deploy.constants as consts


class PlayGo2LocoEnv:
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData

    def __init__(
        self,
        dt=0.02,
        sim_dt=0.002,
        action_scale=0.5,
        headless=False,
    ):
        self.dt = dt
        self.sim_dt = sim_dt
        self.action_scale = action_scale
        self.headless = headless

        self.mj_model = mujoco.MjModel.from_xml_path(consts.FULL_COLLISIONS_FLAT_TERRAIN_XML)
        self.mj_data = mujoco.MjData(self.mj_model)
        # print(len(self.mj_data.ctrl), len(self.mj_data.qpos))
        # raise NotImplementedError
        self.mj_model.opt.timestep = sim_dt
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self._post_init()

    def _post_init(self):
        self.mj_model.dof_damping[6:] = 0.5
        self.mj_model.actuator_gainprm[:, 0] = 35
        self.mj_model.actuator_biasprm[:, 1] = -35

        self._init_q = np.array(consts.DEFAULT_BASE_POSE + consts.DEFAULT_JOINT_ANGLES)
        self._default_pose = np.array(consts.DEFAULT_JOINT_ANGLES)

        self._init_phase = np.array([0, np.pi, np.pi, 0])
        self._stance_phase = np.array([np.pi, np.pi, np.pi, np.pi])

        self._torso_body_id = self.mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self.mj_model.body_subtreemass[self._torso_body_id]

        self._feet_site_id = np.array([self.mj_model.site(name).id for name in consts.FEET_SITES])
        self._floor_geom_id = self.mj_model.geom("floor").id
        self._feet_geom_id = np.array([self.mj_model.geom(name).id for name in consts.FEET_GEOMS])
        self._imu_site_id = self.mj_model.site("imu").id

    def reset(self, base_pose=None, qpos=None):
        if base_pose is None:
            base_pose = np.zeros(7)
            base_pose[:2] = np.random.uniform(-0.2, 0.2, size=2)
            base_pose[2] = consts.DEFAULT_BASE_POSE[2]
            base_pose[3] = 1
            # base_pose[6] = 1
        if qpos is None:
            qpos = self._default_pose
        # self.mj_data.qpos[:2] = np.random.uniform(-0.2, 0.2, size=2)
        # self.mj_data.qpos[2] = consts.DEFAULT_BASE_POSE[2]
        # self.mj_data.qpos[7:] = self._default_pose
        self.mj_data.qpos[:7] = base_pose
        self.mj_data.qpos[7:] = qpos
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        if not self.headless:
            self.viewer.sync()

        gait_freq = 2.0
        phase_dt = 2 * np.pi * self.dt * gait_freq
        foot_height = 0.05

        info = {
            "step": 0,
            "command": np.zeros(3),
            "last_command": np.zeros(3),
            "last_act": np.zeros(self.mj_model.nu),
            "phase": self._init_phase.copy(),
            "phase_dt": phase_dt,
            "foot_height": foot_height,
        }
        obs = self.get_obs(info)
        return info, obs

    def step(self, info, action: np.ndarray, return_all=False):
        # debug
        mujoco.mj_forward(self.mj_model, self.mj_data)
        if not self.headless:
            self.viewer.sync()

        quad_poses = []
        quad_qposes = []
        # debug
        motor_targets = self._default_pose + action * self.action_scale
        self.mj_data.ctrl[:] = motor_targets
        for _ in range(int(self.dt / self.sim_dt)):
            mujoco.mj_step(self.mj_model, self.mj_data)
            quad_poses.append(self.mj_data.qpos[:7])
            quad_qposes.append(self.mj_data.qpos[7:])
        if not self.headless:
            self.viewer.sync()
        if not self.headless:
            time.sleep(self.dt)

        phase_tp1 = info["phase"] + info["phase_dt"]
        info["phase"] = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
        info["step"] += 1
        info["last_last_act"] = info["last_act"].copy()
        info["last_act"] = action.copy()
        obs = self.get_obs(info)
        if return_all:
            return info, obs, quad_poses, quad_qposes
        return info, obs

    def get_obs(self, info):
        # pose
        _linvel = self.get_sensor_data(consts.LOCAL_LINVEL_SENSOR)
        gyro = self.get_sensor_data(consts.GYRO_SENSOR)
        gvec = self.mj_data.site_xmat[self._imu_site_id].reshape(3, 3).T @ np.array([0, 0, -1])

        # joint
        joint_angles = self.mj_data.qpos[7:]
        joint_vel = self.mj_data.qvel[6:]

        gait_phase = np.hstack([np.cos(info["phase"]), np.sin(info["phase"])])
        state = np.hstack(
            [
                # commands
                info["command"],  # 3
                info["foot_height"],
                gait_phase,  # 8
                # pose state
                gyro,  # 3
                gvec,  # 3
                # joint state
                (joint_angles - self._default_pose),  #
                joint_vel,  #
                # history
                info["last_act"],  #
            ]
        )
        return {
            "state": state,
            "privileged_state": None,
        }

    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """Gets sensor data given sensor name."""
        sensor_id = self.mj_model.sensor(sensor_name).id
        sensor_adr = self.mj_model.sensor_adr[sensor_id]
        sensor_dim = self.mj_model.sensor_dim[sensor_id]
        return self.mj_data.sensordata[sensor_adr : sensor_adr + sensor_dim]

    def close(self):
        if not self.headless:
            self.viewer.close()
