import argparse
from tqdm import trange

from src.type import Grasp
from src.sim.grasp_env import Obs, GraspEnvConfig, GraspEnv, get_grasps


def main():
    parser = argparse.ArgumentParser(description="Trajectory Evaluation - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.1)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--wait_steps", type=int, default=25)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--grasp", type=int, default=0)
    args = parser.parse_args()

    for _ in trange(args.num):
        env_config = GraspEnvConfig(
            robot=args.robot,
            obj_name=args.obj,
            headless=args.headless,
            ctrl_dt=args.ctrl_dt,
            wait_steps=args.wait_steps,
        )

        env = GraspEnv(env_config)
        env.launch()
        env.reset()
        obs = env.get_obs()
        env.save_obs(obs)
        if not args.grasp:
            env.close()
            continue

        grasps = get_grasps(args.obj)
        for obj_frame_grasp in grasps:
            robot_frame_grasp = Grasp(
                trans=obs.object_pose[:3, :3] @ obj_frame_grasp.trans
                + obs.object_pose[:3, 3],
                rot=obs.object_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            plan = env.plan_grasp(robot_frame_grasp, obs.robot_frame_pc)
            if plan is not None:
                break

        if plan is not None:
            succ = env.execute_plan(plan)
            print(f"Execution {'succeeded' if succ else 'failed'}")
        else:
            print("No plan found")

        env.close()


if __name__ == "__main__":
    main()
