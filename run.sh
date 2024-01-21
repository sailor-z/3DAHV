# CUDA_VISIBLE_DEVICES=0 python train_estimator.py
# python train_estimator_pl.py
# python train_regressor_objaverse.py

for i in $(seq 1 1 50)
do
   python test_mv_co3d.py
done
