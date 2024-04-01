from types import SimpleNamespace
import wandb
import yaml
import os
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
import lightning.pytorch as pl
from utils import mk_folders
from modules.PL_delta_rota_att_mask_tiny import training_lm, Estimator
# from modules.PL_delta_rota_att_mask_tiny_3d import training_lm, Estimator
# from modules.PL_delta_rota_att_mask_wo_att_tiny import training_lm, Estimator
# from modules.PL_delta_rota_att_mask_img_tiny import training_lm, Estimator


def main(cfg):
    cfg["RUN_NAME"] = 'Objaverse_delta_rota_att_mask'
    cfg["DATA"]["ACC_THR"] = 15
    cfg["DATA"]["NUM_ROTA"] = 9000 #15000
    cfg["TRAIN"]["BS"] = 24#12
    cfg["TRAIN"]["LR"] = 1e-5
    cfg["TRAIN"]["MAX_EPOCH"] = 10
    cfg["DATA"]["BG"] = True
    cfg["TRAIN"]["BG_RATIO"] = 0.5

    cfg["TRAIN"]["MASK"] = True
    cfg["TRAIN"]["MASK_RATIO"] = 0.25

    if cfg["RUN_NAME"] == 'Objaverse_delta_rota_att_wo_mask':
        cfg["TRAIN"]["MASK"] = False
        cfg["TRAIN"]["MASK_RATIO"] = 0.0

    # ##LINEMOD
    # cfg["LINEMOD"]["OCC"] = False

    ##LINEMOD-O
    cfg["LINEMOD"]["OCC"] = True

    print(cfg)

    if cfg["LINEMOD"]["OCC"] == False:
        filename = "checkpoint_lm"
    else:
        filename = "checkpoint_lm_occ"

    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', dirpath=os.path.join("./models", cfg["RUN_NAME"]), filename=filename)

    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="ddp_find_unused_parameters_true", accumulate_grad_batches=1,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"], sync_batchnorm=True, limit_train_batches=cfg["TRAIN"]["SAMPLE_RATE"],
        callbacks=[checkpoint_callback])

    if trainer.global_rank == 0:
        mk_folders(cfg["RUN_NAME"])
        wandb.login(key='b57ba9e1ca7deed19c1ee2b694c16b1744807f4f')
        wandb.init(project="train_delta_rota_att_mask_lm", group="train")

    training_lm(cfg, trainer)

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
