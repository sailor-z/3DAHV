import sys
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
import os
from fastprogress import progress_bar
import lightning.pytorch as pl
from pytorch3d.transforms import random_rotations
from modules.model import Estimator

from data_loader import Dataset_Loader_LINEMOD_stereo as Dataset_Loader
from utils import to_cuda, get_calibration_matrix_K_from_blender, rotate_volume, visualization

CATEGORY_LM = ["CAT", "BENCHVISE", "CAM", "DRILLER", "DUCK"]

def test_category(cfg, model, clsID):
    K = np.array(cfg["LINEMOD"]["INTERNAL_K"]).reshape(3, 3)

    test_dataset = Dataset_Loader(cfg, clsID)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["WORKERS"], drop_last=False)

    bbox_3d = np.asarray(test_dataset.bbox_3d[int(clsID) - 1])

    pbar = progress_bar(test_dataloader, leave=False)

    pred_Rs, pred_errs = [], []
    for idx, data in enumerate(pbar):
        data = to_cuda(data)
        mask_src, mask_tgt = data["src_mask"], data["ref_mask"]
        img_src, img_tgt = data["src_img"], data["ref_img"]
        R_src, R_tgt = data["src_R"], data["ref_R"]
        T_src, T_tgt = data["src_T"], data["ref_T"]
        crop_params_src, crop_params_tgt = data["src_crop_params"], data["ref_crop_params"]

        if torch.any(mask_src.flatten(1).sum(dim=-1) < cfg["DATA"]["SIZE_THR"]) or torch.any(mask_tgt.flatten(1).sum(dim=-1) < cfg["DATA"]["SIZE_THR"]):
            print("Skip bad case")
            continue

        codebook = random_rotations(model.num_rota).to(img_src.device)

        img_feat_src, img_feat_tgt, gt_src_2_tgt_R, gt_tgt_2_src_R = model(img_src, mask_src, R_src, img_tgt, mask_tgt, R_tgt)

        B, C, D, H, W = img_feat_src.shape

        img_feat_src_2_tgt = [rotate_volume(img_feat[None].expand(codebook.shape[0], -1, -1, -1, -1), codebook) for img_feat in img_feat_src]
        img_feat_src_2_tgt = torch.stack(img_feat_src_2_tgt).reshape(-1, C, D, H, W)

        img_feat_src_2_tgt = model.feature_aligner.forward_3d2d(img_feat_src_2_tgt).reshape(B, codebook.shape[0], -1, H*W)

        img_feat_src_2_tgt_gt = rotate_volume(img_feat_src, gt_src_2_tgt_R)
        img_feat_src_2_tgt_gt = model.feature_aligner.forward_3d2d(img_feat_src_2_tgt_gt)

        img_feat_tgt = model.feature_aligner.forward_3d2d(img_feat_tgt)

        pred_sim = (img_feat_src_2_tgt * img_feat_tgt[:, None]).sum(dim=2).mean(dim=-1)
        gt_sim = (img_feat_src_2_tgt_gt * img_feat_tgt).sum(dim=1).mean(dim=-1)

        pred_index = torch.max(pred_sim, dim=1)[1]
        pred_src_2_tgt_R = codebook[pred_index]

        ### geo_dis
        sim = (torch.sum(pred_src_2_tgt_R.view(-1, 9) * gt_src_2_tgt_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi

        pred_errs.append(geo_dis)

        pbar.comment = "Error: %.2f" % (geo_dis.mean().item())

        pred_Rs.append(pred_src_2_tgt_R.cpu().detach().numpy().reshape(-1))

    pred_err = torch.cat(pred_errs)

    pred_acc_30 = 100 * (pred_err < 30).float().mean().item()
    pred_acc_15 = 100 * (pred_err < 15).float().mean().item()
    pred_err = pred_err.mean().item()

    pred_Rs = np.asarray(pred_Rs)
    np.savetxt(os.path.join("./models", cfg["RUN_NAME"], "linemod_pred_Rs_%06d.txt" % (clsID)), pred_Rs)

    print("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (pred_err, pred_acc_30, pred_acc_15))

    return pred_err, pred_acc_30, pred_acc_15

def main(cfg):
    cfg["RUN_NAME"] = 'LINEMOD_3DAHV'
    cfg["DATA"]["OBJ_SIZE"] = 256
    cfg["DATA"]["BG"] = True
    cfg["DATA"]["NUM_ROTA"] = 50000

    print(cfg)

    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], "checkpoint_lm.ckpt")

    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path)
        model.eval()
    else:
        raise RuntimeError("Pretrained model cannot be not found, please check")

    errs, pred_accs_30, pred_accs_15 = [], [], []
    for obj in CATEGORY:
        with torch.no_grad():
            err, pred_acc_30, pred_acc_15 = test_category(cfg, model, cfg["LINEMOD"][obj])

        with open(os.path.join("models", cfg["RUN_NAME"], 'linemod_result.txt'), 'a') as f:
            f.write("%s avg_err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f \n" % \
                (obj, err, pred_acc_30, pred_acc_15))
        f.close()

        errs.append(err)
        pred_accs_30.append(pred_acc_30)
        pred_accs_15.append(pred_acc_15)

    err = np.asarray(errs).mean()
    pred_acc_30 = np.asarray(pred_accs_30).mean()
    pred_acc_15 = np.asarray(pred_accs_15).mean()

    print("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (err, pred_acc_30, pred_acc_15))

    with open(os.path.join("models", cfg["RUN_NAME"], 'linemod_result.txt'), 'a') as f:
        f.write("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f \n" % (err, pred_acc_30, pred_acc_15))
    f.close()

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
