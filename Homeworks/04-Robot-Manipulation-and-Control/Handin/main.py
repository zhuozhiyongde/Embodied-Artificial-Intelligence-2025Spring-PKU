import argparse
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from pyapriltags import Detector

from extra.model import get_model
from extra.path import get_exp_config_from_checkpoint
from extra.utils import get_pc, get_pc_world, get_workspace_mask
from src.config import Config
from src.sim.wrapper_env import WrapperEnv, WrapperEnvConfig, get_grasps

# from src.real.wrapper_env import WrapperEnvConfig, WrapperEnv
# from src.real.wrapper_env import get_grasps
from src.test.load_test import load_test_data
from src.type import Grasp
from src.utils import to_pose


def detect_driller_pose(img, depth, camera_matrix, camera_pose, args):
    """
    Detects the pose of driller, you can include your policy in args
    """

    # Get the full point cloud
    depth = np.array(depth, dtype=np.float32)
    full_pc_camera = get_pc(depth, camera_matrix)
    full_pc_world = (
        np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera) + camera_pose[:3, 3]
    )

    # Extract driller point cloud from the full point cloud
    driller_pc_mask = get_workspace_mask(full_pc_world)
    driller_pc_idx = np.random.randint(0, np.sum(driller_pc_mask), 1024)
    driller_pc_camera = full_pc_camera[driller_pc_mask][driller_pc_idx].astype(
        np.float32
    )
    driller_pc_world = full_pc_world[driller_pc_mask][driller_pc_idx].astype(np.float32)

    # Load model
    config = Config.from_yaml(get_exp_config_from_checkpoint(args.checkpoint))
    model = get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.eval().to(args.device)

    pc_camera_tensor = (
        torch.from_numpy(driller_pc_camera).float().unsqueeze(0).to(args.device)
    )

    with torch.no_grad():
        est_trans, est_rot = model.est(pc_camera_tensor)
        est_trans = est_trans.cpu().numpy()[0]
        est_rot = est_rot.cpu().numpy()[0]

    pose = to_pose(est_trans, est_rot)
    pose = camera_pose @ pose

    return pose


def detect_marker_pose(
    detector: Detector,
    img: np.ndarray,
    camera_params: tuple,
    camera_pose: np.ndarray,
    tag_size: float = 0.12,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detections = detector.detect(
        gray_img, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size
    )

    if not detections:
        return None, None

    detection_result = detections[0]

    pose_R_cam_tag = detection_result.pose_R
    pose_t_cam_tag = detection_result.pose_t

    T_cam_tag = to_pose(trans=pose_t_cam_tag.squeeze(), rot=pose_R_cam_tag)
    T_world_tag = camera_pose @ T_cam_tag

    trans_marker_world = T_world_tag[:3, 3]
    rot_marker_world = T_world_tag[:3, :3]

    return trans_marker_world, rot_marker_world


def forward_quad_policy(pose, target_pose, *args, **kwargs):
    """guide the quadruped to position where you drop the driller"""

    _target_pose = np.linalg.inv(pose) @ target_pose

    x_diff = _target_pose[0, 3]
    y_diff = _target_pose[1, 3]

    sin = _target_pose[1, 0]
    cos = _target_pose[0, 0]
    theta = np.arctan2(sin, cos)
    if theta > 0:
        omega = -0.05
    else:
        omega = 0.05

    v_x = np.clip(np.abs(x_diff) * 0.1, 0.01, 0.15) * np.sign(x_diff)
    v_y = np.clip(np.abs(y_diff) * 0.2, 0.01, 0.15) * np.sign(y_diff)

    action = np.array([v_y, v_x, omega])

    return action


def backward_quad_policy(pose, target_pose, *args, **kwargs):
    """guide the quadruped back to its initial position"""
    # implement

    _target_pose = np.linalg.inv(pose) @ target_pose

    x_diff = _target_pose[0, 3]
    y_diff = _target_pose[1, 3]

    sin = _target_pose[1, 0]
    cos = _target_pose[0, 0]
    theta = np.arctan2(sin, cos)
    if theta > 0:
        omega = -0.01
    else:
        omega = 0.01

    v_x = np.clip(np.abs(x_diff) * 0.3, 0.01, 0.15) * np.sign(x_diff)
    v_y = np.clip(np.abs(y_diff), 0.01, 0.15) * np.sign(y_diff)
    if np.abs(x_diff) * 5 < np.abs(y_diff):
        v_x = 0.0

    action = np.array([v_y, v_x, omega])

    return action


def plan_grasp(
    env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs
) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""

    ## Configurations
    reach_steps = grasp_config["reach_steps"]
    lift_steps = grasp_config["lift_steps"]
    delta_dist = grasp_config["delta_dist"]
    init_qpos = kwargs.get("init_qpos", env.sim.humanoid_robot_cfg.joint_init_qpos)
    obs_wrist = kwargs.get("obs_wrist", None)
    environment_pc = get_pc_world(
        depth=obs_wrist.depth,
        camer_pose=obs_wrist.camera_pose,
        intrinsics=env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics,
    )

    ## Point cloud postprocessing
    while True:
        # Select a random point from the point cloud
        random_index = np.random.randint(0, environment_pc.shape[0])
        selected_pt = environment_pc[random_index]

        # Get the point cloud of the table
        near_mask = np.abs(environment_pc[:, 2] - selected_pt[2]) < 0.01
        near_count = np.sum(near_mask)
        if near_count > 100000 and selected_pt[2] > 0.5:
            environment_pc = environment_pc[near_mask]
            dist_to_origin_xy = np.linalg.norm(environment_pc[:, :2], axis=1)
            dist_mask = dist_to_origin_xy >= 0.3
            environment_pc = environment_pc[dist_mask]
            break

    random_indices = np.random.choice(environment_pc.shape[0], size=1000, replace=False)
    sampled_pc = environment_pc[random_indices]

    ## Solve qpos with inverse kinematics
    # Pregrasp
    pre_grasp_trans = grasp.trans - delta_dist * grasp.rot[:, 0]
    pre_grasp_rot = grasp.rot
    success_pre_grasp, q_pre_grasp = env.humanoid_robot_model.ik(
        pre_grasp_trans, pre_grasp_rot, init_qpos=init_qpos
    )
    if not success_pre_grasp:
        print("IK failed for pre-grasp pose")
        return None

    # Grasp
    grasp_trans = grasp.trans
    grasp_rot = grasp.rot
    success_grasp, q_grasp = env.humanoid_robot_model.ik(
        grasp_trans, grasp_rot, init_qpos=q_pre_grasp
    )
    if not success_grasp:
        print("IK failed for grasp pose")
        return None

    # Lift
    lift_trans = grasp.trans + [0.15, -0.1, 0.1]
    lift_rot = grasp.rot
    success_lift, q_lift = env.humanoid_robot_model.ik(
        lift_trans, lift_rot, init_qpos=q_grasp
    )
    if not success_lift:
        print("IK failed for lift pose")

    ## Motion Planning
    # Reach
    traj_reach = []
    traj_reach = plan_move_qpos(q_pre_grasp, q_grasp, steps=reach_steps)

    # for q_cur in traj_reach:
    #     collided, _ = env.humanoid_robot_model.check_collision(q_cur, pc=sampled_pc)
    #     if collided:
    #         print("Collision in reach plan")
    #         return None

    # Lift
    q_mid = q_grasp.copy()
    q_mid[2] = q_lift[2]
    traj_lift = []
    traj_lift.append(plan_move_qpos(q_grasp, q_mid, steps=lift_steps // 2))
    traj_lift.append(plan_move_qpos(q_mid, q_lift, steps=lift_steps // 2))
    traj_lift = np.concatenate(traj_lift, axis=0)

    # for q_cur in traj_lift:
    #     collided, _ = env.humanoid_robot_model.check_collision(q_cur, pc=sampled_pc)
    #     if collided:
    #         print("Collision in lift plan")
    #         break

    return [traj_reach, traj_lift]


def plan_move(
    env: WrapperEnv, begin_qpos, container_trans, end_rot, steps=50, *args, **kwargs
):
    """Plan a trajectory moving the driller from table to dropping position"""

    _x = -0.1
    _y = 0.05

    success_move = False
    end_trans = container_trans + np.array([_x, _y, 0.60])
    success_move, q_move_1 = env.humanoid_robot_model.ik(
        end_trans, end_rot, init_qpos=begin_qpos
    )
    if success_move:
        print(f"Phase 1: IK succeeded for driller moving to position: {_x}, {_y}")
    else:
        print("IK failed for driller moving after retries")
        return None

    # success_move = False
    # end_trans = container_trans + np.array([_x, _y, 0.30])
    # theta_ys = [-65, -60, -55, -50, -45]
    # for theta_y in theta_ys:
    #     theta_y *= np.pi / 180
    #     rot_y = np.array(
    #         [
    #             [np.cos(theta_y), 0, np.sin(theta_y)],
    #             [0, 1, 0],
    #             [-np.sin(theta_y), 0, np.cos(theta_y)],
    #         ]
    #     )
    #     end_rot = end_rot @ rot_y
    #     success_move, q_move_2 = env.humanoid_robot_model.ik(
    #         end_trans, end_rot, init_qpos=q_move_1
    #     )
    #     if success_move:
    #         print(
    #             f"Phase 2: IK succeeded for driller moving to position: {theta_y}, {_x}, {_y}"
    #         )
    #         break
    # if not success_move:
    #     print("IK failed for driller moving after retries")
    #     return None

    traj = []
    traj.append(plan_move_qpos(begin_qpos, q_move_1, steps=steps // 2))
    # traj.append(plan_move_qpos(q_move_1, q_move_2, steps=steps // 2))
    return np.concatenate(traj, axis=0)


def open_gripper(env: WrapperEnv, steps=10):
    for _ in range(steps):
        env.step_env(gripper_open=1)


def close_gripper(env: WrapperEnv, steps=10):
    for _ in range(steps):
        env.step_env(gripper_open=0)


def plan_move_qpos(begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []

    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())

    return np.array(traj)


def execute_plan(
    env: WrapperEnv,
    plan,
    detector=None,
    target_container_pose=None,
    head_camera_params=None,
):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        if not detector is None:
            obs_head = env.get_obs(camera_id=0)
            trans_marker_world, rot_marker_world = detect_marker_pose(
                detector,
                obs_head.rgb,
                head_camera_params,
                obs_head.camera_pose,
                tag_size=0.12,
            )
            trans_container_world = (
                rot_marker_world @ np.array([0, -0.31, 0.02]) + trans_marker_world
            )
            rot_container_world = rot_marker_world
            pose_container_world = to_pose(trans_container_world, rot_container_world)
            quad_command = forward_quad_policy(
                pose_container_world, target_container_pose
            )
            env.step_env(humanoid_action=plan[step], quad_command=quad_command)
            # print(env.get_container_pose()[:3, 3])
        else:
            env.step_env(humanoid_action=plan[step])


TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = False


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./extra/checkpoint/checkpoint_40000.pth",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )

    env = WrapperEnv(env_config)
    if TESTING:
        data_dict = load_test_data(args.test_id)
        env.set_table_obj_config(
            table_pose=data_dict["table_pose"],
            table_size=data_dict["table_size"],
            obj_pose=data_dict["obj_pose"],
        )
        env.set_quad_reset_pos(data_dict["quad_reset_pos"])

    env.launch()
    env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    Metric = {
        "obj_pose": False,
        "drop_precision": False,
        "quad_return": False,
    }

    head_init_qpos = np.array(
        [0.0, 0.0]
    )  # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)

    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        forward_steps = 1500
        steps_per_camera_shot = 5
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (
            head_camera_matrix[0, 0],
            head_camera_matrix[1, 1],
            head_camera_matrix[0, 2],
            head_camera_matrix[1, 2],
        )

        initial_marker_pose = None
        target_container_pose = np.array(
            [[0, -1, 0, 0.80], [-1, 0, 0, -0.18], [0, 0, -1, 0.385], [0, 0, 0, 1]]
        )

        def is_close(pose1, pose2, threshold):
            dist_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
            rot_diff = pose1[:3, :3] @ pose2[:3, :3].T
            angle_diff = np.abs(np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)))
            if dist_diff < threshold and angle_diff < threshold:
                return True
            return False

        for step in range(forward_steps):
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0)
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector,
                    obs_head.rgb,
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12,
                )
                if trans_marker_world is not None:
                    trans_container_world = (
                        rot_marker_world @ np.array([0, -0.31, 0.02])
                        + trans_marker_world
                    )
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(
                        trans_container_world, rot_container_world
                    )
                    if step == 0:
                        initial_marker_pose = to_pose(
                            trans_marker_world, rot_marker_world
                        )

            quad_command = forward_quad_policy(
                pose_container_world, target_container_pose
            )
            move_head = True
            if move_head:
                head_qpos = [-0.05, 0.3 + 0.4 * step / forward_steps]
                env.step_env(humanoid_head_qpos=head_qpos, quad_command=quad_command)

                # Record initial marker pose for later use
                if initial_marker_pose is None:
                    obs_head = env.get_obs(camera_id=0)
                    trans_marker_world, rot_marker_world = detect_marker_pose(
                        detector,
                        obs_head.rgb,
                        head_camera_params,
                        obs_head.camera_pose,
                        tag_size=0.12,
                    )
                    initial_marker_pose = to_pose(trans_marker_world, rot_marker_world)
            else:
                env.step_env(quad_command=quad_command)

            if is_close(pose_container_world, target_container_pose, threshold=0.025):
                print(f"Quadruped reached the target position at step {step}.")
                break

    # --------------------------------------step 2: detect driller pose------------------------------------------------------

    if not DISABLE_GRASP:
        # Original observing qpos for observation
        observing_qpos = humanoid_init_qpos + np.array([0.01, 0, 0.25, 0, 0, 0, 0.15])
        init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps=200)
        execute_plan(
            env, init_plan, detector, target_container_pose, head_camera_params
        )
        obs_wrist = env.get_obs(camera_id=1)  # wrist camera
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics

        est_driller_pose = detect_driller_pose(
            rgb, depth, wrist_camera_matrix, camera_pose, args
        )
        print(f"Driller pose detected: {est_driller_pose}")

        # metric judgement
        Metric["obj_pose"] = env.metric_obj_pose(est_driller_pose)

        gt_driller_pose = env.get_driller_pose()
        print(f"Ground truth driller pose: {gt_driller_pose}")
        driller_pose = est_driller_pose
        dist_diff = np.linalg.norm(driller_pose[:3, 3] - gt_driller_pose[:3, 3])
        rot_diff = driller_pose[:3, :3] @ gt_driller_pose[:3, :3].T
        angle_diff = np.abs(np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)))
        print(
            f"Distance difference: {dist_diff:.4f}, Rotation difference: {angle_diff:.4f}"
        )

    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------

    if not DISABLE_GRASP:
        observing_qpos = humanoid_init_qpos + np.array([0.01, 0, 0, 0, 0, 0, 0])
        init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps=200)
        execute_plan(
            env, init_plan, detector, target_container_pose, head_camera_params
        )

        def rot_dist(rot1, rot2):
            rot_diff = rot1 @ rot2.T
            angle_diff = np.abs(np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)))
            return angle_diff

        # Grasp configurations
        obj_pose = driller_pose.copy()
        grasps = get_grasps(args.obj)
        grasps0_s = Grasp(
            grasps[0].trans, grasps[0].rot @ np.diag([1, -1, -1]), grasps[0].width
        )
        grasps0_n = Grasp(
            grasps[0].trans, grasps[0].rot @ np.diag([-1, -1, 1]), grasps[0].width
        )
        grasps2_n = Grasp(
            grasps[2].trans, grasps[2].rot @ np.diag([-1, -1, 1]), grasps[2].width
        )
        valid_grasps = [grasps0_s, grasps[0], grasps0_n]
        grasp_config = dict(
            reach_steps=100,
            lift_steps=100,
            delta_dist=0.2,
        )

        # Choose the grasp with the smallest rotation distance
        _, effector_rot = env.humanoid_robot_model.fk_eef(observing_qpos)
        best_ind = 0
        min_rot_diff = 10
        for i, obj_frame_grasp in enumerate(valid_grasps):
            robot_frame_grasp = Grasp(
                trans=obj_pose[:3, :3] @ obj_frame_grasp.trans + obj_pose[:3, 3],
                rot=obj_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            rot_diff = rot_dist(effector_rot, robot_frame_grasp.rot)
            if rot_diff < min_rot_diff:
                best_ind = i
                min_rot_diff = rot_diff
        print(f"Best grasp index: {best_ind}, rotation distance: {min_rot_diff}")
        obj_frame_grasp = valid_grasps[best_ind]
        robot_frame_grasp = Grasp(
            trans=obj_pose[:3, :3] @ obj_frame_grasp.trans + obj_pose[:3, 3],
            rot=obj_pose[:3, :3] @ obj_frame_grasp.rot,
            width=obj_frame_grasp.width,
        )

        # Grasp planning
        obs_wrist = env.get_obs(camera_id=1)
        obs_wrist.rgb = obs_wrist.rgb
        obs_wrist.depth = obs_wrist.depth
        grasp_plan = plan_grasp(
            env,
            robot_frame_grasp,
            grasp_config,
            init_qpos=observing_qpos,
            obs_wrist=obs_wrist,
        )
        if grasp_plan is None:
            print("No valid grasp plan found.")
            env.close()
            return
        reach_plan, lift_plan = grasp_plan

        # Grasp execution
        pregrasp_plan = plan_move_qpos(observing_qpos, reach_plan[0], steps=50)
        execute_plan(
            env, pregrasp_plan, detector, target_container_pose, head_camera_params
        )
        open_gripper(env)
        execute_plan(
            env, reach_plan, detector, target_container_pose, head_camera_params
        )
        close_gripper(env)
        execute_plan(
            env, lift_plan, detector, target_container_pose, head_camera_params
        )

    # --------------------------------------step 4: plan to move and drop----------------------------------------------------

    print("Start moving and dropping the driller...")
    if not DISABLE_GRASP and not DISABLE_MOVE:
        # implement your moving plan
        #
        theta_x = np.pi / 180 * 180
        theta_y = np.pi / 180 * 0
        theta_z = np.pi / 180 * 45

        rot_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)],
            ]
        )
        rot_y = np.array(
            [
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)],
            ]
        )
        rot_z = np.array(
            [
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1],
            ]
        )
        end_rot = rot_x @ rot_y @ rot_z
        move_plan = plan_move(
            env=env,
            begin_qpos=lift_plan[-1],
            container_trans=pose_container_world[:3, 3],
            end_rot=end_rot,
            steps=200,
        )
        execute_plan(
            env, move_plan, detector, target_container_pose, head_camera_params
        )
        for step in range(10):
            env.step_env()
        open_gripper(env)

        # Move the gripper back to lift position to avoid the arm from blocking the marker
        reversed_move_plan = move_plan[::-1]
        execute_plan(
            env, reversed_move_plan, detector, target_container_pose, head_camera_params
        )

    # --------------------------------------step 5: move quadruped backward to initial position------------------------------

    print("Start moving quadruped backward to initial position...")
    if not DISABLE_MOVE:
        backward_steps = 2000
        cur_head_qpos = [-0.05, 0.7]

        def is_close(pose1, pose2, threshold):
            dist_diff = np.max(np.abs(pose1[:2, 3] - pose2[:2, 3]))
            rot_diff = pose1[:3, :3] @ pose2[:3, :3].T
            angle_diff = np.abs(np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1, 1)))
            if dist_diff < threshold and angle_diff < threshold:
                return True
            return False

        for step in range(backward_steps):
            if (step + 1) % 200 == 0:
                print(f"Step {step + 1}/{backward_steps}")

            for i in range(100):
                obs_head = env.get_obs(camera_id=0)
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector,
                    obs_head.rgb,
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12,
                )
                cur_marker_pose = to_pose(trans_marker_world, rot_marker_world)
                if not np.allclose(cur_marker_pose, np.eye(4)):
                    break
                cur_head_qpos[1] -= 3e-3
                env.step_env(
                    humanoid_head_qpos=cur_head_qpos,
                )

            # If the marker is detected, use the backward policy to return to the initial position
            quad_command = backward_quad_policy(cur_marker_pose, initial_marker_pose)
            env.step_env(humanoid_head_qpos=cur_head_qpos, quad_command=quad_command)

            if is_close(cur_marker_pose, initial_marker_pose, threshold=0.02):
                print(f"Quadruped returned to the initial position at step {step}.")
                break

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric)

    print("Simulation completed.")
    env.close()


if __name__ == "__main__":
    main()
