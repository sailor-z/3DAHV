# 3D-Aware Hypothesis & Verification for Generalizable Relative Object Pose Estimation
PyTorch implementation of "3D-Aware Hypothesis & Verification for Generalizable Relative Object Pose Estimation" (ICLR 2024)

[[project page](https://sailor-z.github.io/projects/ICLR2024_3DAHV.html)] &nbsp; &nbsp; &nbsp; &nbsp; [[paper](https://arxiv.org/pdf/2310.03534.pdf)]

# Setup Dependencies
```
conda create -n 3DAHV python=3.9
conda activate 3DAHV
bash ./install.sh
```

# Experiments on CO3D

## Data Preparation
Please refer to the instructions provided in [RelPose++](https://github.com/amyxlase/relpose-plus-plus/tree/main?tab=readme-ov-file#pre-processing-co3d) for downloading and preprocessing CO3D. If necessary, you may need to adjust the values for `["CO3D"]["CO3D_DIR"]` and `["CO3D"]["CO3D_ANNOTATION_DIR"]` in the `config.yaml` file to match the actual directory path of your data.

## Test pretrained model
We provide a model pretrained on the training set of CO3D. Please download it [here](https://drive.google.com/file/d/1lxVnY8o3_pzGqejJO4kLO7OV7R3XnHw8/view?usp=sharing). We store this pretrained model at `./models/Co3d_3DHAV` by default.
Run the following evaluation to get the results:
```
python ./test_co3d.py
```
Notably, the reproduced results might be slightly different from those reported in the paper. This is because the image pairs during testing are randomly sampled in the RelPose++ implementation.

## Train model on CO3D
Run the following script to train the model:
```
python ./train_estimator_co3d.py
```
In our experiments, we trained the model on two A100 GPUs, and it took about two days to complete the training process.

# Todo list
- [ ] We're in the process of cleaning up the code. If you find any bugs, don't hesitate to reach out. It would be greatly appreciated.
- [ ] The evaluation code on Objaverse and LINEMOD will be released soon. Stay tuned.

# Citation
If you find the project useful, please consider citing:
```bibtex
@article{zhao20233d,
    title={3D-Aware Hypothesis \& Verification for Generalizable Relative Object Pose Estimation},
    author={Zhao, Chen and Zhang, Tong and Salzmann, Mathieu},
    journal={Proceedings of the International Conference on Learning Representations},
    year={2024}
  }
}
```
