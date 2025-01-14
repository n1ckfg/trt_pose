DIR=$PWD

cd ..

rm -rf pytorch-install
mkdir pytorch-install
cd pytorch-install

# https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048

# TORCH
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install numpy cython
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl 

# TORCHVISION
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
export BUILD_VERSION=0.11.1  # where 0.x.0 is the torchvision version  
python3 setup.py install --user
cd ..

# JETCAM
git clone https://github.com/NVIDIA-AI-IOT/jetcam
cd jetcam
sudo python3 setup.py install
cd ..

# TORCH2TRT
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install --plugins

cd $DIR
sudo apt-get install python3-matplotlib libhdf5-dev libffi-dev
pip3 install tqdm pycocotools 
# jupyter notebook
pip3 install pandas pillow==4.1.1
#pip3 install h5py scipy scikit-image scikit-learn 
pip3 install cffi jupyter ipywidgets packaging notebook jupyterlab

sudo python3 setup.py install


