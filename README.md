<div align="center">

# TBDVO

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description
The implementation of the approach from the FRUCT Conference Paper "Transformer-Based Deep Monocular Visual Odometry for Edge Devices"

## Prepare data

Create folder `data`. 

Download KITTI dataset for odometry into `data/kitti_dataset`. 

Download pretrained FlowNet weights (flownets_bn_EPE2.459.pth.tar) in `data/checkpoints` from [repository](https://github.com/ClementPinard/FlowNetPytorch/blob/master/README.md). 

## How to run
Install dependencies
```shell
# clone project
git clone https://github.com/toshiks/TBDVO.git
cd TBDVO

# create conda environment
conda env create -f conda_env_gpu.yaml -n myenv
conda activate myenv
```

Train model with default configuration (deepvo original)

```shell
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```shell
python run.py experiment=experiment_name
```

You can override any parameter from command line like this
```shell
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

Run benchmarks:
```shell
export PYTHONPATH=$PWD
python util_scripts/benchmarks.py
```

<br>
