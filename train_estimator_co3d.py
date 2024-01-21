from types import SimpleNamespace
import yaml
import os
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
import lightning.pytorch as pl
from utils import mk_folders
from modules.model_co3d import training

def main(cfg):
    cfg["RUN_NAME"] = 'Co3d_3DHAV'
    cfg["DATA"]["ACC_THR"] = 15
    cfg["DATA"]["NUM_ROTA"] = 9000
    cfg["TRAIN"]["MASK"] = True
    cfg["TRAIN"]["MASK_RATIO"] = 0.25
    cfg["TRAIN"]["BS"] = 32
    cfg["TRAIN"]["LR"] = 1e-4

    cfg["CO3D"]["CO3D_DIR"] = "/scratch/cvlab/datasets/common/co3d/data"
    cfg["CO3D"]["CO3D_ANNOTATION_DIR"] = "/scratch/cvlab/datasets/common/co3d/preprocessed"

    print(cfg)

    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', dirpath=os.path.join("./models", cfg["RUN_NAME"]), filename='checkpoint_co3d')

    ### multiple GPUs
    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="ddp_find_unused_parameters_true", accumulate_grad_batches=1,
        max_epochs=250, sync_batchnorm=True, callbacks=[checkpoint_callback])

    training(cfg, trainer)

if __name__ == '__main__':
    with open("./config.yaml", 'r') as load_f:
        cfg = yaml.load(load_f, Loader=yaml.FullLoader)
    load_f.close()

    main(cfg)
