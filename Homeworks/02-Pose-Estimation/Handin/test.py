import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import get_model
from src.config import Config
from src.path import get_exp_config_from_checkpoint
from src.data import PoseDataset
from src.utils import rot_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="val")
    args = parser.parse_args()

    # load config
    config = Config.from_yaml(get_exp_config_from_checkpoint(args.checkpoint))

    # load model
    model = get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.eval().to(args.device)

    # load dataset
    dataset = PoseDataset(config, mode=args.mode)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    trans_dists, rot_dists = [], []

    with torch.set_grad_enabled(False):
        for data in tqdm(dataloader):
            data = {k: v.to(args.device) for k, v in data.items()}
            est_trans, est_rot = model.est(data["pc"])
            assert torch.allclose(
                torch.bmm(est_rot, est_rot.transpose(1, 2)),
                torch.eye(3, device=args.device),
                atol=1e-5,
            ), "Estimated rotation matrix is not orthogonal"
            assert (
                est_rot.det() > 0
            ).all(), "Estimated rotation matrix is not a valid rotation matrix"
            gt_trans, gt_rot = data["trans"], data["rot"]
            trans_dists.append((est_trans - gt_trans).norm(dim=-1).cpu().numpy())
            est_rot, gt_rot = est_rot.cpu().numpy(), gt_rot.cpu().numpy()
            for i in range(len(est_rot)):
                rot_dists.append(rot_dist(est_rot[i], gt_rot[i]))
    trans_dists = np.concatenate(trans_dists)
    rot_dists = np.array(rot_dists)
    print(f"Error: trans {np.mean(trans_dists)} rot {np.mean(rot_dists)}")


if __name__ == "__main__":
    main()
