# BALANCED STRIPE-WISE PRUNING IN THE FILTER (BSWP)
![image](https://github.com/ajdt1111/BSWP/blob/main/framework.png)
## Introduction
    This work has been accepted at ICASSP 2022.
## Requirements
python3.6 <br>
pytorch1.8.0 <br>
torchvision0.9.0 <br>
## Getting Started
#### Training and pruning VGG with piecewise function
    python main_vgg.py --arch VGG --save checkpoint/VGG --use_function piecewise
#### Training and pruning VGG with continuous function
    python main_vgg.py --arch VGG --save checkpoint/VGG --use_function continuous
#### Training and pruning ResNet with piecewise function
    python main_resnet.py --arch ResNet56 --save checkpoint/ResNet56 --use_function piecewise
#### Training and pruning ResNet with continuous function
    python main_resnet.py --arch ResNet56 --save checkpoint/ResNet56 --use_function continuous
## References
Our work is based on : Pruning Filter in Filter (https://arxiv.org/abs/2009.14410) <br>
Our code is alse based on : https://github.com/fxmeng/Pruning-Filter-in-Filter <br>
