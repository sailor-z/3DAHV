sudo apt-get update -y
sudo apt-get install ffmpeg libsm6 libxext6 -y
sudo apt-get install libopenexr-dev -y
sudo apt-get install openexr -y

conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
conda install -c fastai fastprogress -y
conda install -c conda-forge pyyaml -y

yes | pip install --upgrade pip
yes | pip install path
yes | pip install yaml
yes | pip install tensorboardX
yes | pip install imutils
yes | pip install pandas
yes | pip install opencv-python
yes | pip install scikit-image
yes | pip install timm==0.6.12
yes | pip install matplotlib
yes | pip install diffusers transformers scipy ftfy accelerate
yes | pip install einops
yes | pip install --upgrade --quiet objaverse
yes | pip install trimesh
yes | pip install antialiased-cnns
yes | pip install lightning
yes | pip install -U scikit-learn
