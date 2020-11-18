# Implementation of SSDLite in PyTorch 1.2+ 

This is the object detection code for our [MobileNeXt](https://arxiv.org/pdf/2007.02269.pdf) paper.
This repository implements [SSDLite](https://arxiv.org/abs/1512.02325), which is presented in [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf). 
The implementation is heavily influenced by the [SSD](https://github.com/lufficc/SSD) project.

## Highlights

- **PyTorch 1.2**: Support PyTorch 1.2 or higher.
- **Multi-GPU training and inference**: We use `DistributedDataParallel`, you can train or test with arbitrary GPU(s), the training schema will change accordingly.
- **Better results than the original TF version**: We achieve an mAP score of 22.3 on COCO, which is slightly better than the original TF version when taking MobileNetV2 as backbone. 
- **Implementation of SSDLite with FPN**: Support SSDLite with FPN, which highly improves the baseline results.

## Experiment Setup

Please refer to [INSTALL.md](https://github.com/Andrew-Qibin/ssdlite-pytorch/blob/master/INSTALL.md) for all the information about installation.

For training on COCO, the batch size is set to 256 and the initial learning rate is set to 0.01. We use 8 GPUs with 12 GB memory to run the experiments
with synchronized batch normalization (important). You can also use less GPUs as long as the GPU memory is enough but make sure that the batch size is 256.
For example, 4 V100 GPUs are also enough for running.

For training on Pascal VOC, the batch size is set to 24 and the initial learning rate is set to 0.001. We use 4 GPUs with 12 GB memory to run the experiments
with the standard batch normalization.

For more implementation details, please refer to the configs in this project.

## MODEL ZOO

### Pretrained Models:
|   Networks   |     Links    | 
| :----------: | :----------: |
|  MobilenetV2 |  [model](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) |
|  MobileNeXt  |  [model](https://github.com/Andrew-Qibin/ssdlite-pytorch/blob/master/weights/mnext.pth.tar) |

### COCO:

| Backbone                   | Input Size   |       Box AP     | Model Size | Download  |
| :------------------------: | :----------: | :--------------: | :--------: | :-------: |
|  MobileNetV2 (SSDLite)     |     320      |          22.3    | 34M        | [model](https://drive.google.com/file/d/1jKa16d2c7zSrIYzKAVb9PyAQH2ifXCih/view?usp=sharing)   |
|  MobileNeXt (SSDLite)      |     320      |          23.3    | 36M        | [model](https://drive.google.com/file/d/1GlBU-10YBjGhj9snw9JVU6i__dc9-INY/view?usp=sharing)   |

### PASCAL VOC:

| Backbone               | Input Size  |          mAP         | Model Size | Download  |
| :--------------------: | :----------:|   :--------------:   | :--------: | :-------: |
|  VGG16 (SSD)           |     300     |          77.7        |   201MB    | [model](https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd300_voc0712.pth)  |
|  VGG16 (SSDLite)       |     512     |          80.7        |   207MB    | [model](https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd512_voc0712.pth)  |
|  MobilenetV2 (SSDLite) |     320     |          71.8        |   26MB     | [model](https://drive.google.com/file/d/1rWVxlWGeCylH-sz43PKQUlOsWTzxZTnq/view?usp=sharing) |
|  MobileNeXt (SSDLite)  |     320     |          72.6        |   27MB     | [model](https://drive.google.com/file/d/1s365AwRVdGMGDZrSjMN58qOx7ydmbutx/view?usp=sharing) |



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
