import os

import matplotlib.pyplot as plt

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import time
import logging

_start = time.time()


class ETFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return f"{record.created - _start:.1f}s"


fmt = "%(asctime)s [%(levelname)s] - %(message)s"
handler = logging.StreamHandler()
handler.setFormatter(ETFormatter(fmt))

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)

import jax
from jax import numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1.getup import Getup

DESIRED_BODY_HEIGHT = 0.33


class MyGetupEnv(Getup):
    def step(self, state: mjx_env.State, action: jax.Array):
        motor_targets = state.data.qpos[7:] + action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        # --- Key State Extraction ---

        # Unit vector pointing up in world frame (z-axis negative in many physics engines)
        up_vec = jp.array([0.0, 0.0, -1.0])

        # Extract the 3D position of the IMU site (typically mounted on the body), in world coordinates [m]
        body_pos = data.site_xpos[self._imu_site_id]

        # Compute the linear velocity of the body in its local (body) frame [m/s]
        body_lin_vel = self.get_local_linvel(data)

        # Retrieve the angular velocity (gyroscope data) of the body in its local frame [rad/s]
        body_ang_vel = self.get_gyro(data)

        # Get the gravity vector expressed in the local body frame, usually used to infer orientation
        gravity_vector = self.get_gravity(data)

        # Make a copy of the default pose (e.g., reference or rest configuration) of the full model
        default_qpos = self._default_pose.copy()

        # Extract joint positions (excluding root pose: typically first 7 elements are base position + orientation)
        joint_qpos = data.qpos[7:]

        # Extract joint velocities (excluding root velocities)
        joint_qvel = data.qvel[7:]

        # TODO: your code here.
        #  Hint: consider three import objective related to getup task:
        #   1. body height
        #   2. body orientation
        #   3. joint position (error to default pose)
        #   4. body angular velocity (error to zero)
        scale_factors = [20.0, 5.0, 0.1, 0.1]
        height_scale, orientation_scale, joint_scale, ang_vel_scale = scale_factors

        cur_height = body_pos[2]
        height_error_sq = jp.square(cur_height - DESIRED_BODY_HEIGHT)
        rew_height = jp.exp(-height_scale * height_error_sq)

        gravity_norm = jp.linalg.norm(gravity_vector)
        gravity_norm = jp.where(gravity_norm < 1e-6, 1e-6, gravity_norm)
        gravity_vector = gravity_vector / gravity_norm
        orientation_error_sq = jp.square(gravity_vector[2] + 1.0)
        rew_orientation = jp.exp(-orientation_scale * orientation_error_sq)

        joint_pos_error_sq = jp.mean(jp.square(joint_qpos - default_qpos))
        rew_joint_pos = jp.exp(-joint_scale * joint_pos_error_sq)

        ang_vel_error_sq = jp.sum(jp.square(body_ang_vel))
        rew_ang_vel = jp.exp(-ang_vel_scale * ang_vel_error_sq)

        reward_weight = jp.array([3.0, 1.0, 2.0, 0.3])
        # reward_weight = jp.array([2.0, 1.2, 0.3, 0.1])
        reward_array = jp.array(
            [rew_height, rew_orientation, rew_joint_pos, rew_ang_vel]
        )
        reward = jp.sum(reward_weight * reward_array)
        # TODO: End of your code.

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        state.metrics["reward"] = reward
        done = jp.float32(done)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Sample a random initial configuration with some probability.
        # rng, key1, key2 = jax.random.split(rng, 3)
        # qpos = jp.where(
        #     jax.random.bernoulli(key1, self._config.drop_from_height_prob),
        #     self._get_random_qpos(key2),
        #     self._init_q,
        # )
        qpos = self._init_q.copy()
        # Sample a random root velocity.
        # rng, key = jax.random.split(rng)
        qvel = jp.zeros(self.mjx_model.nv)
        # qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Let the robot settle for a few steps.
        data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
        data = data.replace(time=0.0)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
        }

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        metrics = {"reward": jp.zeros(())}
        return mjx_env.State(data, obs, reward, done, metrics, info)


def create_env():
    # env config
    env_cfg = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        Kp=35.0,
        Kd=0.5,
        episode_length=300,
        drop_from_height_prob=0.6,
        settle_time=0.5,
        action_repeat=1,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        energy_termination_threshold=np.inf,
        noise_config=config_dict.create(
            level=0.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
            ),
        ),
    )
    env = MyGetupEnv(env_cfg)
    return env


def train_ppo():
    import mediapy as media

    from datetime import datetime
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from ml_collections import config_dict

    ppo_params = config_dict.create(
        num_timesteps=40_000_000,
        num_evals=0,
        reward_scaling=1.0,
        episode_length=500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=5e-4,  # 3e-4
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=128,  # 256
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="privileged_state",
            value_obs_key="privileged_state",
        ),
    )

    start_t = datetime.now()

    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        # progress_fn=progress,
        num_eval_envs=0,
        log_training_metrics=True,
        training_metrics_steps=1_000_000,
    )

    env = create_env()
    eval_env = create_env()
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    end_t = datetime.now()
    print(f"time to train: {end_t - start_t}")

    render_length = 1000
    _pre_render_length = 100

    # Enable perturbation in the eval env.
    eval_env = create_env()

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(0)
    rollout = []
    body_height = []

    # is_upright = self._is_upright(gravity)
    # is_at_desired_height = self._is_at_desired_height(torso_height)
    # gate = is_upright * is_at_desired_height

    state = jit_reset(rng)
    for i in range(render_length):
        if i < _pre_render_length:
            ctrl = env._default_pose.copy()
        else:
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)

        state = jit_step(state, ctrl)
        rollout.append(state)
        env_height = state.data.site_xpos[env._imu_site_id][2]
        body_height.append(env_height)

    body_height = jp.array(body_height)
    height_error = np.mean(np.abs(body_height - DESIRED_BODY_HEIGHT))
    plt.plot(body_height)
    # plot desired body height
    plt.axhline(DESIRED_BODY_HEIGHT, color="r", linestyle="--")
    plt.title(f"Height error: {height_error:.3f}")
    plt.xlabel("steps")
    plt.ylabel("body height")
    plt.savefig("part2_height_error.png")

    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    print(f"fps: {fps}")

    traj = rollout[::render_every]
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    frames = eval_env.render(
        traj,
        camera="track",
        height=480,
        width=640,
        scene_option=scene_option,
    )
    media.write_video("../experiments/solutions/part2_video.mp4", frames)
    print("video saved to part2.mp4")


if __name__ == "__main__":
    train_ppo()
