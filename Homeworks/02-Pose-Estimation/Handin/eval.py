import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.type import Grasp
from src.model import get_model
from src.config import Config
from src.path import get_exp_config_from_checkpoint
from src.sim.grasp_env import Obs, GraspEnvConfig, GraspEnv, get_grasps
from src.data import PoseDataset
from src.utils import transform_grasp_pose
from src.vis import Vis


def main():
    parser = argparse.ArgumentParser(description="Trajectory Evaluation - Physics")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.1)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--wait_steps", type=int, default=15)
    parser.add_argument("--try_plan_num", type=int, default=3)
    parser.add_argument("--vis", type=int, default=1)
    args = parser.parse_args()

    # load config & dataset
    config = Config.from_yaml(get_exp_config_from_checkpoint(args.checkpoint))
    dataset = PoseDataset(config, mode=args.mode)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # load model
    model = get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.eval().to(args.device)

    result = []
    for dic in tqdm(dataloader):
        dic = {k: v[0].numpy() for k, v in dic.items()}
        robot_frame_pc = (
            np.einsum("ab,nb->na", dic["camera_pose"][:3, :3], dic["pc"])
            + dic["camera_pose"][:3, 3]
        )
        object_pose = dic["obj_pose_in_world"]

        env_config = GraspEnvConfig(
            robot=args.robot,
            obj_name=args.obj,
            headless=args.headless,
            ctrl_dt=args.ctrl_dt,
            wait_steps=args.wait_steps,
            obj_pose=object_pose,
        )

        with torch.no_grad():
            est_trans, est_rot = model.est(
                torch.from_numpy(dic["pc"])[None].to(args.device)
            )
            est_trans, est_rot = est_trans[0].cpu().numpy(), est_rot[0].cpu().numpy()

        if args.vis:
            Vis.show(
                Vis.pc(dic["pc"])
                + Vis.mesh(
                    os.path.join("asset", "obj", args.obj, "single.obj"),
                    trans=est_trans,
                    rot=est_rot,
                    opacity=0.8,
                )
            )

        env = GraspEnv(env_config)
        env.launch()
        env.reset()
        grasps = get_grasps(args.obj)
        for obj_frame_grasp in grasps:
            robot_frame_grasp = transform_grasp_pose(
                obj_frame_grasp,
                est_trans,
                est_rot,
                dic["camera_pose"][:3, 3],
                dic["camera_pose"][:3, :3],
            )
            for _ in range(args.try_plan_num):
                plan = env.plan_grasp(robot_frame_grasp, robot_frame_pc)
                if plan is not None:
                    break
            if plan is not None:
                break

        if plan is not None:
            succ = env.execute_plan(plan)
            print(f"Execution {'succeeded' if succ else 'failed'}")
        else:
            succ = False
            print("No plan found")
        result.append(succ)
        print(
            f"Current success rate: {sum(result)}/{len(result)} = {sum(result) / len(result)}"
        )

        env.close()


if __name__ == "__main__":
    main()
