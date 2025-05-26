from refiner.refiner import PoseRefiner
from pytorch3d.io import IO
import torch
import numpy as np


def refine_pose(model_path, sample, pred_r, pred_t, num_iters=50):
    gt_depth = sample["original_depth"]
    K = np.array(
        [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
    )
    mesh = IO().load_mesh(model_path).to("cuda")
    rf = PoseRefiner(mesh, K, (480, 640))
    R = torch.tensor(pred_r)
    T = torch.tensor(pred_t).squeeze() / 1000
    return rf.refine_pose(R, T, gt_depth.cuda(), num_iters=num_iters, lr=0.0005)
