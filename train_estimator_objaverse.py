from types import SimpleNamespace
import yaml
import os
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
import lightning.pytorch as pl
from utils import mk_folders
from modules.model import training, Estimator

def main(cfg):
    cfg["RUN_NAME"] = 'Objaverse_3DAHV'
    cfg["DATA"]["ACC_THR"] = 15
    cfg["DATA"]["NUM_ROTA"] = 9000
    cfg["TRAIN"]["BS"] = 12
    cfg["TRAIN"]["LR"] = 1e-4
    cfg["DATA"]["BG"] = True
    cfg["TRAIN"]["BG_RATIO"] = 0.5

    cfg["TRAIN"]["MASK"] = True
    cfg["TRAIN"]["MASK_RATIO"] = 0.25

    print(cfg)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc_15', mode='max', dirpath=os.path.join("./models", cfg["RUN_NAME"]), filename='checkpoint_objaverse')

    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="ddp_find_unused_parameters_true", accumulate_grad_batches=1,
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"], sync_batchnorm=True, limit_train_batches=cfg["TRAIN"]["SAMPLE_RATE"],
        callbacks=[checkpoint_callback])

    training(cfg, trainer)

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
