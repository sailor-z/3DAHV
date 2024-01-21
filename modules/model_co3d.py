import sys
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning.pytorch as pl
import numpy as np
from utils import *
from data_loader_co3d import Co3dDataset as Dataset_Loader
from pytorch3d.transforms import random_rotations
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
        self.mid_channel = 256
        self.feature_extractor = DPT_SwinV2_T_256(pretrained=True)
        self.feature_aligner = Feature_Aligner(in_channel=768, mid_channel=256, out_channel=32, n_heads=4, depth=4)

        self.step_outputs = []

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

        loss = -torch.log(positive_sim / positive_negative_sim.clamp(min=1e-8)).mean()

        return loss

    def forward(self, img_src, img_tgt):
        img_feat_src = self.feature_extraction(img_src)
        img_feat_tgt = self.feature_extraction(img_tgt)

        img_feat_src, img_feat_tgt = self.feature_aligner.forward_2d3d(img_feat_src, img_feat_tgt, random_mask=False, mask_ratio=0)

        return img_feat_src, img_feat_tgt

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        img_src = batch["image"][:, 0]
        img_tgt = batch["image"][:, 1]
        gt_src_2_tgt_R = batch["relative_rotation"].squeeze(1)

        img_feat_src = self.feature_extraction(img_src)
        img_feat_tgt = self.feature_extraction(img_tgt)

        img_feat_src, img_feat_tgt = self.feature_aligner.forward_2d3d(img_feat_src, img_feat_tgt,
            random_mask=self.cfg["TRAIN"]["MASK"], mask_ratio=self.cfg["TRAIN"]["MASK_RATIO"])

        B, C, D, H, W = img_feat_src.shape

        with torch.no_grad():
            sampled_R = random_rotations(B*(self.num_rota-1)).to(img_src.device).reshape(B, self.num_rota-1, 3, 3)
            sampled_R = torch.cat([gt_src_2_tgt_R[:, None], sampled_R], dim=1)

        loss = self.infoNCE_loss(img_feat_src, img_feat_tgt, sampled_R, gt_src_2_tgt_R)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW([{"params":self.feature_aligner.parameters(), 'lr':float(self.cfg["TRAIN"]["LR"])},
        {"params":self.feature_extractor.parameters(), 'lr':float(self.cfg["TRAIN"]["LR"])}], eps=1e-5)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        return [optimizer], [scheduler]

def training(cfg, trainer):
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(cfg["DATA"]["OBJ_SIZE"]),
            transforms.Normalize(
                cfg['DATA']['PIXEL_MEAN'],
                cfg['DATA']['PIXEL_STD']),
        ]
    )

    train_dataset = Dataset_Loader(
        cfg=cfg,
        category=["all"],
        split="train",
        transform=trans,
        random_aug=True,
        eval_time=False,
        num_images=2,
        normalize_cameras=False,
        img_size=cfg["DATA"]["OBJ_SIZE"]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["TRAIN"]["BS"], shuffle=True,
        num_workers=cfg["TRAIN"]["WORKERS"], drop_last=True)

    model = Estimator(cfg)
    ckpt_path = os.path.join("models", cfg["RUN_NAME"], 'checkpoint_co3d.ckpt')

    if os.path.exists(ckpt_path):
        print("Loading the pretrained model from the last checkpoint")
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    else:
        print("Train from scratch")
        trainer.fit(model, train_dataloader)
