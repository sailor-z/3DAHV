import sys
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import numpy as np
from pytorch3d.transforms import random_rotations
from utils import *
from data_loader import Dataset_Loader_Objaverse_stereo as Dataset_Loader
from data_loader import Dataset_Loader_Objaverse_stereo_test as Dataset_Loader_Test
from data_loader import Dataset_Loader_LINEMOD_stereo_train as Dataset_Loader_LM

from modules.modules import Feature_Aligner

sys.path.append("./MiDaS")
from hubconf import DPT_SwinV2_T_256
torch.hub.set_dir("./pretrained_models")

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True
torch.autograd.set_detect_anomaly(True)
np.set_printoptions(suppress=True, threshold=2**31-1)
torch.set_float32_matmul_precision("highest")

class Estimator(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_rota = cfg["DATA"]["NUM_ROTA"]
        self.feature_extractor = DPT_SwinV2_T_256(pretrained=True)
        self.feature_aligner = Feature_Aligner(in_channel=768, mid_channel=256, out_channel=32, n_heads=4, depth=4)
        self.step_outputs = []
        self.gt_dis = []
        self.pred_Rs = []

    def feature_extraction(self, img):
        layer_1, layer_2, layer_3, layer_4 = self.feature_extractor.forward_transformer(self.feature_extractor.pretrained, img)
        return layer_4

    def infoNCE_loss(self, img_feat_1, img_feat_2, sampled_R, gt_delta_R):
        bs = gt_delta_R.shape[0]
        with torch.no_grad():
            gt_sim = (torch.sum(sampled_R.flatten(2) * gt_delta_R.view(-1, 1, 9), dim=-1).clamp(-1, 3) - 1) / 2
            gt_dis = torch.arccos(gt_sim) / np.pi
            posi_indices = [torch.nonzero(180 * gt_dis[i] <= self.cfg["DATA"]["ACC_THR"]).squeeze(-1) for i in range(bs)]
            nega_indices = [torch.nonzero(180 * gt_dis[i] > self.cfg["DATA"]["ACC_THR"]).squeeze(-1) for i in range(bs)]

        img_feat_warp = [rotate_volume(img_feat_1[idx:idx+1].expand(self.num_rota, -1, -1, -1, -1), sampled_R[idx]) for idx in range(bs)]

        img_feat_warp = [self.feature_aligner.forward_3d2d(img_feat) for img_feat in img_feat_warp]
        img_feat_2 = self.feature_aligner.forward_3d2d(img_feat_2)

        sim = [(img_feat_warp[idx] * img_feat_2[idx:idx+1]).sum(dim=1).mean(dim=-1) for idx in range(bs)]

        positive_sim = torch.stack([torch.exp(sim[idx][posi_indices[idx]] / 0.1).sum(dim=0) for idx in range(bs)])
        positive_negative_sim = (torch.exp(torch.stack(sim) / 0.1)).sum(dim=-1)

        loss = -torch.log(positive_sim / positive_negative_sim.clamp(min=1e-8))

        return loss

    def forward(self, img_src, mask_src, img_tgt, mask_tgt):
        ### mask the input image
        if self.cfg["DATA"]["BG"] is False:
            img_src = img_src * mask_src
            img_tgt = img_tgt * mask_tgt

        img_feat_src = self.feature_extraction(img_src)
        img_feat_tgt = self.feature_extraction(img_tgt)

        img_feat_src, img_feat_tgt = self.feature_aligner.forward_2d3d(img_feat_src, img_feat_tgt, random_mask=False, mask_ratio=0.0)

        return img_feat_src, img_feat_tgt

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]

        ### mask the input image
        if self.cfg["DATA"]["BG"] is False:
            img_src = img_src * mask_src
            img_tgt = img_tgt * mask_tgt

        with torch.no_grad():
            gt_src_2_tgt_R = torch.bmm(R_tgt, torch.inverse(R_src))
            gt_tgt_2_src_R = torch.bmm(R_src, torch.inverse(R_tgt))

        img_feat_src = self.feature_extraction(img_src)
        img_feat_tgt = self.feature_extraction(img_tgt)

        img_feat_src, img_feat_tgt = self.feature_aligner.forward_2d3d(img_feat_src, img_feat_tgt,
            random_mask=self.cfg["TRAIN"]["MASK"], mask_ratio=self.cfg["TRAIN"]["MASK_RATIO"])

        B, C, D, H, W = img_feat_src.shape

        with torch.no_grad():
            self.Rs  = random_rotations(B*(self.num_rota-1)).to(img_src.device).reshape(B, self.num_rota-1, 3, 3)
            self.Rs = torch.cat([gt_src_2_tgt_R[:, None], self.Rs], dim=1)

        valid = (mask_src.flatten(1).sum(dim=-1) > self.cfg["DATA"]["SIZE_THR"]) * (mask_tgt.flatten(1).sum(dim=-1) > self.cfg["DATA"]["SIZE_THR"])
        if "dis_init" in batch.keys():
            dis_init = batch["dis_init"]
            valid = valid * (dis_init < self.cfg["DATA"]["VIEW_THR"]).float()

        loss = self.infoNCE_loss(img_feat_src, img_feat_tgt, self.Rs, gt_src_2_tgt_R)
        loss = loss * valid
        loss = loss.sum() / valid.sum().clamp(min=1e-8)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]

        img_feat_src, img_feat_tgt = self.forward(img_src, mask_src, img_tgt, mask_tgt)

        with torch.no_grad():
            gt_src_2_tgt_R = torch.bmm(R_tgt, torch.inverse(R_src))
            gt_tgt_2_src_R = torch.bmm(R_src, torch.inverse(R_tgt))

        B, C, D, H, W = img_feat_src.shape

        sampled_R = random_rotations(self.num_rota).to(img_src.device)

        img_feat_src_2_tgt = [rotate_volume(img_feat[None].expand(sampled_R.shape[0], -1, -1, -1, -1), sampled_R) for img_feat in img_feat_src]
        img_feat_src_2_tgt = torch.stack(img_feat_src_2_tgt).reshape(-1, C, D, H, W)
        img_feat_src_2_tgt = self.feature_aligner.forward_3d2d(img_feat_src_2_tgt).reshape(B, self.num_rota, -1, H*W)

        img_feat_src_2_tgt_gt = rotate_volume(img_feat_src, gt_src_2_tgt_R)
        img_feat_src_2_tgt_gt = self.feature_aligner.forward_3d2d(img_feat_src_2_tgt_gt)

        img_feat_tgt = self.feature_aligner.forward_3d2d(img_feat_tgt)

        pred_sim = (img_feat_src_2_tgt * img_feat_tgt[:, None]).sum(dim=2).mean(dim=-1)
        gt_sim = (img_feat_src_2_tgt_gt * img_feat_tgt).sum(dim=1).mean(dim=-1)

        pred_sim, pred_index = torch.max(pred_sim, dim=1)
        pred_src_2_tgt_R = sampled_R[pred_index]

        ### geo_dis
        sim = (torch.sum(pred_src_2_tgt_R.view(-1, 9) * gt_src_2_tgt_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi

        pred_acc_15 = (geo_dis <= 15).float().mean()
        pred_acc_30 = (geo_dis <= 30).float().mean()

        self.log("val_acc_15", pred_acc_15.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc_30", pred_acc_30.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.step_outputs.append(geo_dis)

    def on_validation_epoch_end(self):
        geo_dis = torch.cat(self.step_outputs)

        pred_acc_15 = 100 * (geo_dis <= 15).float().mean()
        pred_acc_30 = 100 * (geo_dis <= 30).float().mean()

        self.step_outputs.clear()

    def test_step(self, batch, batch_idx):
        mask_src, mask_tgt = batch["src_mask"], batch["ref_mask"]
        img_src, img_tgt = batch["src_img"], batch["ref_img"]
        R_src, R_tgt = batch["src_R"], batch["ref_R"]

        if torch.any(mask_src.flatten(1).sum(dim=-1) < self.cfg["DATA"]["SIZE_THR"]) or torch.any(mask_tgt.flatten(1).sum(dim=-1) < self.cfg["DATA"]["SIZE_THR"]):
            print("Skip bad case")
            return 0

        img_feat_src, img_feat_tgt = self.forward(img_src, mask_src, img_tgt, mask_tgt)

        with torch.no_grad():
            gt_src_2_tgt_R = torch.bmm(R_tgt, torch.inverse(R_src))
            gt_tgt_2_src_R = torch.bmm(R_src, torch.inverse(R_tgt))

        B, C, D, H, W = img_feat_src.shape
        sampled_R = random_rotations(self.num_rota).to(img_src.device)

        img_feat_src_2_tgt = [rotate_volume(img_feat[None].expand(sampled_R.shape[0], -1, -1, -1, -1), sampled_R) for img_feat in img_feat_src]
        img_feat_src_2_tgt = torch.stack(img_feat_src_2_tgt).reshape(-1, C, D, H, W)

        img_feat_src_2_tgt = self.feature_aligner.forward_3d2d(img_feat_src_2_tgt).reshape(B, self.num_rota, -1, H*W)

        img_feat_tgt = self.feature_aligner.forward_3d2d(img_feat_tgt)

        pred_sim = (img_feat_src_2_tgt * img_feat_tgt[:, None]).sum(dim=2).mean(dim=-1)

        pred_sim, pred_index = torch.max(pred_sim, dim=1)
        pred_src_2_tgt_R = sampled_R[pred_index]

        ### geo_dis
        sim = (torch.sum(pred_src_2_tgt_R.view(-1, 9) * gt_src_2_tgt_R.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        geo_dis = torch.arccos(sim) * 180. / np.pi

        gt_sim = (torch.sum(R_src.view(-1, 9) * R_tgt.view(-1, 9), dim=-1).clamp(-1, 3) - 1) / 2
        gt_dis = torch.arccos(gt_sim) * 180. / np.pi

        self.step_outputs.append(geo_dis)
        self.gt_dis.append(gt_dis)
        self.pred_Rs.append(pred_src_2_tgt_R.cpu().detach().numpy().reshape(-1))

        self.log("test_error", geo_dis.mean().item(), on_step=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer = optim.AdamW([{"params":self.feature_aligner.parameters(), 'lr':float(self.cfg["TRAIN"]["LR"])},
        {"params":self.feature_extractor.parameters(), 'lr':0.1*float(self.cfg["TRAIN"]["LR"])}], eps=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        return [optimizer], [scheduler]

def training(cfg, trainer):
    val_dataset = Dataset_Loader_Test(cfg, None)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg["TRAIN"]["WORKERS"], drop_last=False)

    train_dataset = Dataset_Loader(cfg, "train", None)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)

    model = Estimator(cfg)
    ckpt_path = os.path.join("models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')
    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader, val_dataloader)

def training_lm(cfg, trainer):
    CATEGORY = ["APE", "CAN", "EGGBOX", "GLUE", "HOLEPUNCHER", "IRON", "LAMP", "PHONE"]
    clsIDs = [cfg["LINEMOD"][cat] for cat in CATEGORY]

    train_dataset = Dataset_Loader_LM(cfg, clsIDs)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)

    model = Estimator(cfg)
    checkpoint_path = os.path.join("./models", cfg["RUN_NAME"], 'checkpoint_objaverse.ckpt')
    if os.path.exists(checkpoint_path):
        print("Loading the pretrained model from " + checkpoint_path)
        model = Estimator.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        raise RuntimeError("Pretrained model cannot be not found, please check")

    filename = "checkpoint_lm.ckpt"

    ckpt_path = os.path.join("models", cfg["RUN_NAME"], filename)
    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader)
