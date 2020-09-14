# Implementation of SSDLite in PyTorch 1.2+


This repository implements [SSDLite (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the projects [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). This repository aims to be the code base for researches based on SSD.

## Highlights

- **PyTorch 1.2**: Support PyTorch 1.2 or higher.
- **Multi-GPU training and inference**: We use `DistributedDataParallel`, you can train or test with arbitrary GPU(s), the training schema will change accordingly.
- **Better results than the original TF version**: We achieve an mAP score of 22.3 on COCO, which is slightly better than the original TF version when taking MobileNetV2 as backbone. 

## Experiment Setup

During training on COCO, the batch size is set to 256 and the initial learning rate is set to 0.01. We use 8 GPUs to run the experiments
with synchronized batch normalization.

## MODEL ZOO

### COCO:

| Backbone       | Input Size  |          box AP                  | Parameters | M-Adds | Model Size |  Download |
| :------------: | :----------:|   :--------------------------:   | :--------: | :-------: | :-------: | :-------: |
|  MobileNetV2 (SSDLite)         |     320     |          22.3  | 4.5M | 0.8B |  262MB     | [model]
|  MobileNeXt (SSDLite)         |     320     |          23.3  | 4.5M | 0.8B |  275MB     | [model]   |

### PASCAL VOC:

| Backbone         | Input Size  |          mAP                     | Model Size | Download  |
| :--------------: | :----------:|   :--------------------------:   | :--------: | :-------: |
|  VGG16           |     300     |          77.7                    |   201MB    | [model](https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd300_voc0712.pth)  |
|  VGG16           |     512     |          80.7                    |   207MB    | [model](https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd512_voc0712.pth)  |
|  Mobilenet V2    |     320     |          68.8                    |   25.5MB   | [model](https://github.com/lufficc/SSD/releases/download/1.2/mobilenet_v2_ssd320_voc0712.pth) |
|  EfficientNet-B3 |     300     |          73.9                    |   97.1MB   | [model](https://github.com/lufficc/SSD/releases/download/1.2/efficient_net_b3_ssd300_voc0712.pth) |



## Citations
If you use this project in your research, please cite this project.
```text
@misc{lufficc2018ssd,
    author = {Congcong Li},
    title = {{High quality, fast, modular reference implementation of SSD in PyTorch}},
    year = {2018},
    howpublished = {\url{https://github.com/lufficc/SSD}}
}
```
