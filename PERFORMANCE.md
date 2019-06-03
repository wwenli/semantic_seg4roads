# Comparison of state-of-the-art Methods for Semantic Segmentation

| Methods | Reported mIoU at PASCAL VOC 2012 | Reported mIoU at Cityscape | Selected Repository | Comments |
| --- | --- | --- | --- | --- |
| FCN | 67.2% | 65.3% | [Keras-FCN](https://github.com/aurora95/Keras-FCN) | Easy to use. More training epochs will improve the performance (may reach at 66%) |
| DeepLabv2 | 79.7% | 70.4% | [DeepLab-ResNet-TensorFlow](https://github.com/DrSleep/tensorflow-deeplab-resnet) | The repository is well written. |
| ResNet-38 | 82.5% | 80.6% | [ademxapp-MXNet](https://github.com/itijyou/ademxapp) | Version is incompatible (MXNet 0.8.0 is needed or the code should be modified) |
| PSPNet | 85.4% | 81.2% | [PSPNet-Keras-tensorflow](https://github.com/jmtatsch/PSPNet-Keras-tensorflow) | Codes and trained parameters for Prediction are avaliable and easy to use. BUT training is HARD (several GPU should be used due to limitation of physical memory on each GPU card). |
| DeepLabv3 | 86.9% | 81.3% | [deeplab3](https://github.com/tensorflow/models/tree/master/research/deeplab) | tensorflow implementation of deeplab3 |
