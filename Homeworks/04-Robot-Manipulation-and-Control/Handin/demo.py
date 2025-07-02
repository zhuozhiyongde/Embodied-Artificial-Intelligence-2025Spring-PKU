import argparse
import numpy as np
import cv2
from pyapriltags import Detector

from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv

def detect_pose(detector, img, camera_matrix, camera_pose):
    """
    Detects the pose of the AprilTag in the image using the given camera matrix and pose.
    detector: AprilTag detector
    img: input image
    camera_matrix: (3, 3) camera intrinsic matrix
    camera_pose: (4, 4) camera pose in world frame
    """
    camera_params = (camera_matrix[0,0],camera_matrix[1,1],camera_matrix[0,2],camera_matrix[1,2])
    pose_marker = np.eye(4)
    # implement the detection logic here
    # 
    return pose_marker

def demo_sim():
    # nearly all the functions of the simulation is implemeted in this demo
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--customize_scene", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    

    args = parser.parse_args()
    
    # april tag is used for detection, tag used in the simulation has size of 0.12
    apriltag_detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )

    env = WrapperEnv(env_config)
    if args.customize_scene:
        # Customize the table and object for testing with random environment
        table_pose = to_pose(trans=np.array([0.6, 0.35, 0.72]))
        table_size = np.array([0.68, 0.36, 0.02])
        obj_trans = np.array([0.5, 0.3, 0.82])
        obj_rot = np.eye(3)
        obj_pose = to_pose(obj_trans, obj_rot)
        env.set_table_obj_config(
            table_pose=table_pose,
            table_size=table_size,
            obj_pose=obj_pose
        )

    env.launch()
    env.reset()

    head_init_qpos = np.array([-0.05, 0.35])  # [horizontal, vertical]
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos

    env.step_env(
        humanoid_head_qpos=head_init_qpos, # head joint qpos is for adjusting the camera pose
        humanoid_action=humanoid_init_qpos[:7],
        quad_command=[0,0,0]
    )
    obs_head = env.get_obs(camera_id=0) # head camera
    obs_wrist = env.get_obs(camera_id=1) # wrist camera
    env.debug_save_obs(obs_head, 'data/obs_head') # obs has rgb, depth, and camera pose
    env.debug_save_obs(obs_wrist, 'data/obs_wrist')

    trans, rot = env.humanoid_robot_model.fk_link(humanoid_init_qpos, env.humanoid_robot_cfg.link_eef)
    succ, qpos = env.humanoid_robot_model.ik(trans=trans, rot=rot)
    if succ:
        print("IK success")
        print("qpos:", qpos)
    
    env.sim.debug_vis_pose(to_pose(trans, rot)) # visualize the pose of the end effector
    for i in range(50):
        env.step_env(
            humanoid_head_qpos=head_init_qpos,
            humanoid_action=humanoid_init_qpos[:7],
            quad_command=[0,0,0]
        )

    humanoid_curr_qpos = humanoid_init_qpos[:7].copy()
    for i in range(100):
        quad_command = np.array([0.3, 0 ,0]) # quad robot move forward
        delta_qpos = np.random.uniform(-0.01, 0.01, size=7) # humanoid robot random action
        humanoid_curr_qpos += delta_qpos
        env.step_env(
            humanoid_head_qpos=head_init_qpos,
            humanoid_action=humanoid_curr_qpos,
            quad_command=quad_command
        )
    
    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    demo_sim()