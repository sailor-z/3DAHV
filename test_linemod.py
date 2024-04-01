import sys
import numpy as np
import cv2
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
import os
from fastprogress import progress_bar
import lightning.pytorch as pl
from pytorch3d.transforms import random_rotations
from modules.PL_delta_rota_att_mask_tiny import Estimator
# from modules.PL_delta_rota_att_mask_wo_att_tiny import Estimator
# from modules.PL_delta_rota_att_mask_tiny_3d import Estimator
# from modules.PL_delta_rota_att_mask_img_tiny import Estimator

from data_loader import Dataset_Loader_LINEMOD_stereo as Dataset_Loader
from utils import to_cuda, get_calibration_matrix_K_from_blender, rotate_volume, visualization

# CATEGORY = ["APE", "BENCHVISE", "CAM", "CAN", "CAT", "DRILLER", "DUCK", "EGGBOX", "GLUE", "HOLEPUNCHER", "IRON", "LAMP", "PHONE"]
CATEGORY_LM = ["CAT", "BENCHVISE", "CAM", "DRILLER", "DUCK"]
CATEGORY_LM_O = ["CAT", "DRILLER", "DUCK"]

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

        # start_time = time.time()

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

        # print("Time consumption:", time.time() - start_time)

        ### geo_dis
        sim = (torch.sum(pred_src_2_tgt_R.view(-1, 9) * gt_src_2_tgt_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi

        # ### detailed result
        # if geo_dis < 10:
        #     sim = (torch.sum(codebook.view(-1, 9) * gt_src_2_tgt_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        #     geo_dis = torch.arccos(sim) * 180. / np.pi
        #
        #     np.savetxt('./visualization/geo_dis.txt', geo_dis.cpu().detach().numpy())
        #     np.savetxt('./visualization/pred_sim.txt', pred_sim[0].cpu().detach().numpy())
        #     exit()



        pred_errs.append(geo_dis)

        pbar.comment = "Error: %.2f" % (geo_dis.mean().item())

        pred_Rs.append(pred_src_2_tgt_R.cpu().detach().numpy().reshape(-1))

        ### viz
        viz = False
        if viz and idx % 30 == 0:
            out_dir = os.path.join("./models", cfg["RUN_NAME"], "visual", "linemod")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            img_src, img_tgt, viz = visualization(cfg, img_src, img_tgt, pred_src_2_tgt_R, R_src, R_tgt, T_tgt, crop_params_tgt, K, bbox_3d)

            cv2.imwrite(os.path.join(out_dir, "%02d_%04d_query.png" % (clsID, idx)), img_tgt)
            cv2.imwrite(os.path.join(out_dir, "%02d_%04d_support.png" % (clsID, idx)), img_src)
            cv2.imwrite(os.path.join(out_dir, "%02d_%04d_%.2f.png" % (clsID, idx, geo_dis.mean().item())), viz)

    pred_err = torch.cat(pred_errs)

    pred_acc_30 = 100 * (pred_err < 30).float().mean().item()
    pred_acc_15 = 100 * (pred_err < 15).float().mean().item()
    pred_err = pred_err.mean().item()

    pred_Rs = np.asarray(pred_Rs)
    np.savetxt(os.path.join("./models", cfg["RUN_NAME"], "linemod_pred_Rs_%06d.txt" % (clsID)), pred_Rs)

    print("err: %.2f || avg_acc_30: %.2f || avg_acc_15: %.2f " % (pred_err, pred_acc_30, pred_acc_15))

    return pred_err, pred_acc_30, pred_acc_15

def main(cfg):
    cfg["RUN_NAME"] = 'Objaverse_delta_rota_att_mask' #'Co3d_delta_rota_att_mask' #
    cfg["DATA"]["OBJ_SIZE"] = 256
    cfg["DATA"]["BG"] = True
    cfg["DATA"]["NUM_ROTA"] = 50000

    ##LINEMOD
    cfg["LINEMOD"]["OCC"] = False
    # ##LINEMOD-O
    # cfg["LINEMOD"]["OCC"] = True

    print(cfg)

    if cfg["LINEMOD"]["OCC"] == False:
        filename = "checkpoint_lm.ckpt"
        CATEGORY = CATEGORY_LM
    else:
        filename = "checkpoint_lm_occ.ckpt"
        CATEGORY = CATEGORY_LM_O

    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], filename)
    # checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], 'checkpoint_lm_200.ckpt')
    # checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')

    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg, img_size=cfg["DATA"]["OBJ_SIZE"])
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

    # for num in [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]:
    #     print("Runing experiments with the number of sampling as:", num)
    #     cfg["DATA"]["NUM_ROTA"] = num
    #     main(cfg)

    # # ### robust to noise
    # cfg["DATA"]["JITTER"] = True
    # for noise in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    #     print("Runing experiments with a noise level of:", noise)
    #     cfg["DATA"]["NOISE_LEVEL"] = noise
    #     main(cfg)
