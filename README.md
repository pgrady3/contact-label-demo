## Installation

```
# Create a new conda environment and install PyTorch. Many ways to accomplish this
conda create -n contact-label python=3.10
conda activate contact-label
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

# Install the segmentation models pytorch project. Must be done by cloning from git
git clone https://github.com/qubvel/segmentation_models.pytorch
pip install -e segmentation_models.pytorch/

# Clone this repo
git clone https://github.com/pgrady3/contact-label-demo.git
cd contact-label-demo
pip install -r requirements.txt
```

The model file can be downloaded from this link. It should be moved to data/model/iccv_rebase_0.pth
```
https://www.dropbox.com/scl/fi/gzqvb0bepx14bc40yy2vh/iccv_rebase_0.pth?rlkey=rjiy9vyp66aiwo8tj45zconha&dl=0
```

## Run webcam demo
```
python -m paper.demo_webcam --config iccv_rebase
```