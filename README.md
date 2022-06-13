# BI475_project
This project is the final project of course BI475. We use models and methods such as R(2+1)D[^1] to classify children's exercise intensity based on short videos. Actually, this project is a video classification problem in video understanding in the context of biomedicine.

## Environment

Before training and testing, you first need to install and determine the dependent environment.

```shell
pip install torch torchvision
pip install tqdm
pip install av==8.0.3
pip install tensorboard
```

The program runs with CUDA on an Nvidia GPU, so the program is best run on the GPU.

## Training

The training file for the model is `train.py`. The specific parameter settings are as follows:

```python
# params
seed = 88
epochs = 100
batch_size = 16
lr = 0.001
weight_decay = 5e-4
dropout = 0.05
device = 'cuda'

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
```

The model type can be selected during training, there are three options: `r3d`, `mc3` and `r2plus1d`. Among them, the memory required for `r2plus1d` training is large, so if it cannot run normally, the parallel processing of the `pytorch` framework should be used. **Of course, a GPU with larger memory is the best choice.**

```python
type = 'r2plus1d'
model = R3DModel(dropout=dropout, type=type).to(device)
# you could add these codes 
from torch.nn import DataParallel
if type == 'r2plus1d':
    model = DataParallel(model)
```

Run command for the training file:

```shell
python train.py
```

## Test

We provide a script file `test.py` for model testing and prediction. We can make predictions on the entire raw dataset, as well as classifying individual video files. The trained models and sample MP4 file are stored in the test folder and can be used directly. You could run the command:

```shell
python test.py
```

If you want to know more information and need to change some parameters, you could run the command:

```shell
python test.py -h
```

You may also encounter memory problems here, and the solution is the same as above.

## Reference

[^1]: Tran, Du, et al. "A closer look at spatiotemporal convolutions for action recognition." *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*. 2018.
