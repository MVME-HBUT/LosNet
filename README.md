## **Introduction**

LosNet is a novel efficient and lightweight framework for ore image segmentation.

Source code for 'Efficient and Lightweight Framework for Real-Time Ore Image Segmentation Based on Deep Learning'. For more details, please refer to our [paper](https://www.mdpi.com/2075-163X/12/5/526).

The source code can be found in [AdelaiDet](https://github.com/aim-uofa/AdelaiDet.git) . AdelaiDet is an open source toolbox for multiple instance-level recognition tasks on top of [Detectron2](https://github.com/facebookresearch/detectron2).

## Installation

### Requirements

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.7 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional but needed by demo and visualization

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

*Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.*

Then build LosNet with:

```
git clone https://github.com/HDL-YD/LosNet.git
cd LosNet
python setup.py build develop
```

If you are using docker, a pre-built image can be pulled with:

```
docker pull tianzhi0549/adet:latest
```

Some projects may require special setup, please follow their own `README.md` in [configs](configs).

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/LosNet/LosNet.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/LosNet
```

To evaluate the model after training, run:

```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/LosNet/LosNet.yaml \
    --eval-only \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/LosNet \
    MODEL.WEIGHTS training_dir/LosNet/model_final.pth
```

## **Run Demo**

```
CUDA_VISIBLE_DEVICES=1 python demo.py \    
	--config-file ../configs/LosNet/LosNet.yaml \     
	--input ../demo \    
	--output ../LosNet/demo_results \   
	--opts MODEL.WEIGHTS ../training_dir/LosNet/model_final.pth
```

<img src="demo\images\1.png"> <img src="demo\images\2.png">

Note that:

- The configs are made for 1-GPU training. To train on another number of GPUs, change the `--num-gpus`.
- If you want to measure the inference time, please change `--num-gpus` to 1.
- We set `OMP_NUM_THREADS=1` by default, which achieves the best speed on our machines, please change it as needed.
- This quick start is made for LosNet. If you are using other projects, please check the projects' own `README.md` in [configs](configs). 
- When you run the demo, the folder is determined according to your situation.

## **Cite**

```
@Article{min12050526,
AUTHOR = {Sun, Guodong and Huang, Delong and Cheng, Le and Jia, Junjie and Xiong, Chenyun and Zhang, Yang},
TITLE = {Efficient and Lightweight Framework for Real-Time Ore Image Segmentation Based on Deep Learning},
JOURNAL = {Minerals},
VOLUME = {12},
YEAR = {2022},
NUMBER = {5},
ARTICLE-NUMBER = {526},
}
```
