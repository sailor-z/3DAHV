"""
Script for pairwise evaluation of predictor (ie, given 2 images, compute accuracy of
highest scoring mode).

Note that here, num_frames refers to the number of images sampled from the sequence.
The input frames will be all NP2 permutations of using those image frames for pairwise
evaluation.
"""

import argparse
import sys
import os
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm
from torchvision import transforms
from pytorch3d.transforms import random_rotations
from modules.PL_delta_rota_att_mask_tiny_co3d import Estimator
from data_loader_co3d import Co3dDataset
from data_loader_co3d import TEST_CATEGORIES
from utils import to_cuda, rotate_volume

torch.manual_seed(0)
np.random.seed(0)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--use_pbar", action="store_true")
    return parser

def compute_angular_error(rotation1, rotation2):
    R_rel = rotation1.T @ rotation2
    tr = (np.trace(R_rel) - 1) / 2
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi


def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi

def get_permutations(num_frames):
    permutations = []
    for i in range(num_frames):
        for j in range(num_frames):
            if i != j:
                permutations.append((i, j))
    return torch.tensor(permutations)


def get_dataset(cfg=None, category="banana", split="train", dataset="co3d"):
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(cfg["DATA"]["OBJ_SIZE"]),
            transforms.Normalize(
                cfg['DATA']['PIXEL_MEAN'],
                cfg['DATA']['PIXEL_STD']),
        ]
    )

    if dataset == "co3dv1":
        return Co3dv1Dataset(
            cfg=cfg,
            split=split,
            category=[category],
            transform=trans,
            random_aug=False,
            normalize_cameras=False,
            eval_time=True,
            img_size=cfg["DATA"]["OBJ_SIZE"]
        )
    elif dataset in ["co3d", "co3dv2"]:
        return Co3dDataset(
            cfg=cfg,
            split=split,
            category=[category],
            transform=trans,
            random_aug=False,
            eval_time=True,
            normalize_cameras=False,
            img_size=cfg["DATA"]["OBJ_SIZE"]
        )
    else:
        raise Exception(f"Unknown dataset {dataset}")


def evaluate_category(
    cfg,
    model,
    category="banana",
    split="train",
    num_frames=2,
    use_pbar=False,
    dataset="co3d",
):
    dataset = get_dataset(cfg=cfg, category=category, split=split, dataset=dataset)
    device = "cuda"

    permutations = get_permutations(num_frames)
    proposals = random_rotations(cfg["DATA"]["NUM_ROTA"]).to(device)
    angular_errors = []
    iterable = tqdm(dataset) if use_pbar else dataset
    for metadata in iterable:
        n = metadata["n"]
        sequence_name = metadata["model_id"]
        key_frames = np.random.choice(n, num_frames, replace=False)
        batch = dataset.get_data(sequence_name=sequence_name, ids=key_frames)

        images = batch["image"]
        images_permuted = images[permutations].to(device)

        rotations = batch["R"]
        rotations_permuted = rotations[permutations].to(device)

        rotations_gt = torch.bmm(
            rotations_permuted[:, 0].transpose(1, 2),
            rotations_permuted[:, 1],
        )

        for i in range(len(permutations)):
            image1 = images_permuted[i, 0]
            image2 = images_permuted[i, 1]
            R1 = rotations_permuted[i, 0]
            R2 = rotations_permuted[i, 1]
            gt_src_2_tgt_R = rotations_gt[i]

            img_feat_src, img_feat_tgt = model(image1[None], image2[None])

            B, C, D, H, W = img_feat_src.shape

            img_feat_src_2_tgt = [rotate_volume(img_feat[None].expand(proposals.shape[0], -1, -1, -1, -1), proposals) for img_feat in img_feat_src]
            img_feat_src_2_tgt = torch.stack(img_feat_src_2_tgt).reshape(-1, C, D, H, W)

            img_feat_src_2_tgt = model.feature_aligner.forward_3d2d(img_feat_src_2_tgt).reshape(B, proposals.shape[0], -1, H*W)
            img_feat_tgt = model.feature_aligner.forward_3d2d(img_feat_tgt)

            pred_sim = (img_feat_src_2_tgt * img_feat_tgt[:, None]).sum(dim=2).mean(dim=-1)

            pred_sim, pred_index = torch.max(pred_sim, dim=1)
            pred_src_2_tgt_R = proposals[pred_index]

            ### geo_dis
            sim = (torch.sum(pred_src_2_tgt_R.view(-1, 9) * gt_src_2_tgt_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
            err = torch.arccos(sim) * 180. / np.pi

            angular_errors.append(err.mean().item())

    return np.array(angular_errors)


def evaluate_pairwise(
    cfg=None,
    model=None,
    split="train",
    num_frames=2,
    print_results=True,
    use_pbar=False,
    categories=TEST_CATEGORIES,
    dataset="co3d",
):
    errors = {}
    errors_15 = {}
    errors_30 = {}
    for category in categories:
        angular_errors = evaluate_category(
            cfg=cfg,
            model=model,
            category=category,
            split=split,
            num_frames=num_frames,
            use_pbar=use_pbar,
            dataset=dataset,
        )
        errors[category] = np.mean(angular_errors)
        errors_15[category] = 100*np.mean(angular_errors < 15)
        errors_30[category] = 100*np.mean(angular_errors < 30)

        print(category + " err: %.2f || acc_30: %.2f || acc_15: %.2f " % (errors[category], errors_30[category], errors_15[category]))

    errors["mean"] = np.mean(list(errors.values()))
    errors_15["mean"] = np.mean(list(errors_15.values()))
    errors_30["mean"] = np.mean(list(errors_30.values()))
    if print_results:
        print(f"{'Category':>10s}{'<15':6s}{'<30':6s}")
        for category in errors_15.keys():
            print(
                f"{category:>10s}{errors_15[category]:6.02f}{errors_30[category]:6.02f}"
            )

    print("avg_err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (errors["mean"], errors_30["mean"], errors_15["mean"]))

    return errors, errors_30, errors_15


if __name__ == "__main__":
    args = get_parser().parse_args()

    args.num_frames = 2
    args.use_pbar = True

    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    cfg["RUN_NAME"] = "Co3d_3DHAV"
    cfg["DATA"]["NUM_ROTA"] = 50000

    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], 'checkpoint_co3d.ckpt')

    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg)
        model.eval()
    else:
        raise RuntimeError("Pretrained model cannot be not found, please check")

    errors, errors_30, errors_15 = {}, {}, {}
    for i in range(5):
        error, error_30, error_15, error_hist = evaluate_pairwise(cfg=cfg,
            model=model,
            num_frames=args.num_frames,
            print_results=True,
            use_pbar=args.use_pbar,
            split="test",
        )
        if i == 0:
            for category in error.keys():
                errors[category] = []
                errors_30[category] = []
                errors_15[category] = []

        for category in error.keys():
            errors[category].append(error[category])
            errors_30[category].append(error_30[category])
            errors_15[category].append(error_15[category])

    for category in errors.keys():
        errors[category] = np.asarray(errors[category]).mean()
        errors_30[category] = np.asarray(errors_30[category]).mean()
        errors_15[category] = np.asarray(errors_15[category]).mean()

    print(f"{'Category':>10s}{'<15':6s}{'<30':6s}")
    with open(os.path.join("models", cfg["RUN_NAME"], 'co3d_result.txt'), 'a') as f:
        for category in errors_15.keys():
            print(f"{category:>10s}{errors[category]:6.02f}{errors_15[category]:6.02f}{errors_30[category]:6.02f}")
            f.write(f"{category:>10s}{errors[category]:6.02f}{errors_15[category]:6.02f}{errors_30[category]:6.02f} \n")
    f.close()
