# Implementation of SSDLite in PyTorch 1.2+ (Object detection code for our [MobileNeXt](https://arxiv.org/pdf/2007.02269.pdf) paper)


This repository implements [SSDLite](https://arxiv.org/abs/1512.02325), which is presented in [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf). 
The implementation is heavily influenced by the projects [SSD](https://github.com/lufficc/SSD).

## Highlights

- **PyTorch 1.2**: Support PyTorch 1.2 or higher.
- **Multi-GPU training and inference**: We use `DistributedDataParallel`, you can train or test with arbitrary GPU(s), the training schema will change accordingly.
- **Better results than the original TF version**: We achieve an mAP score of 22.3 on COCO, which is slightly better than the original TF version when taking MobileNetV2 as backbone. 
- **Implementation of SSDLite with FPN**: Support SSDLite with FPN, which highly improves the baseline results.

## Experiment Setup

During training on COCO, the batch size is set to 256 and the initial learning rate is set to 0.01. We use 8 GPUs to run the experiments
with synchronized batch normalization. For more implementation details, please refer to the configs in this project.

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
@inproceedings{daquan2020rethinking,
  title={Rethinking Bottleneck Structure for Efficient Mobile Network Design},
  author={Daquan, Zhou and Hou, Qibin and Chen, Yunpeng and Feng, Jiashi and Yan, Shuicheng},
  booktitle={European conference on computer vision},
  year={2020}
}
```

```text
@misc{hou2020ssdite-pytorch,
    author = {Qibin Hou},
    title = {{Fast Implementation of SSDLite in PyTorch}},
    year = {2020},
    howpublished = {\url{https://github.com/Andrew-Qibin/ssdlite-pytorch}}
}
```
