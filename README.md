## Introduction
A ReID framework implemented by PyTorch.


## Features
- **Backbone**: ResNet, DenseNet, MGN(TBD), PCB(TBD)   
- **Loss**: Softmax + CrossEntropyLoss(label smooth), TripletLoss
    
- **Distance**: Cos, Euclid
    
- **Augmentation**: Flip, Crop, Erase
- **Warm-up**
- **Re-Rank**
    
    
## Datasets
- [X] Market1501
- [X] Your own dataset.


## Performance on Market1501
+ Loss: softmax + CELoss, Lr: 0.05, BatchSize: 32, Epoch: 60, Dist: Cos

| Model | LabelSmooth | ReRank | WarmUp | RE | @Rank1(%) | mAP(%) | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet 50 |     |     |   |     | 89.1 | 72.6 | [download](https://pan.baidu.com/s/1apUwOtnCp6JYKf73nUkxNg) |
| ResNet 50 |     |     |   | Yes | 90.1 | 75.7 | [download](https://pan.baidu.com/s/1eMbTmwE9LcdPkUt-0NA70Q) |
| ResNet 50 |     |     | 5 |     | 88.6 | 73.0 | [download](https://pan.baidu.com/s/1JYaNkrn08VJihEakAivI8g) |
| ResNet 50 |     |     | 20 |    | 87.9 | 72.8 |  |
| ResNet 50 |     | Yes |   |     | 90.6 | 85.5 | [download](https://pan.baidu.com/s/1apUwOtnCp6JYKf73nUkxNg) |
| ResNet 50 | Yes |     |   |     | 88.6 | 72.0 |  |
| ResNet 50 |     |     | 5 | Yes | 90.3 | 76.0 | [download](https://pan.baidu.com/s/1UHDQb8UTiyogJ6zBSMF_sA) |
| ResNet 50 |     | Yes | 5 | Yes | 92.4 | 89.3 | [download](https://pan.baidu.com/s/1UHDQb8UTiyogJ6zBSMF_sA) |
| ResNet 50 | Yes |     | 5 | Yes | 91.3 | 78.7 | [download](https://pan.baidu.com/s/16jMkqdmYhAMSPgGB7OLzMw) |
| ResNet 50 | Yes | Yes | 5 | Yes | 92.7 | 89.9 | [download](https://pan.baidu.com/s/16jMkqdmYhAMSPgGB7OLzMw) |
| ResNet 50 | Yes |     | 20 | Yes | 91.6 | 79.0 | [download]() |
| ResNet 50 | Yes | Yes | 20 | Yes | 93.2 | 90.5 | [download]() |

+ Loss: softmax + CELoss, Lr: 0.05, BatchSize: 32, Epoch: 60, **Dist: Euclid**

| Model | LabelSmooth | ReRank | WarmUp | RE | @Rank1(%) | mAP(%) | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet 50 |     |     |   |     | 88.4 | 71.0 | [download](https://pan.baidu.com/s/1apUwOtnCp6JYKf73nUkxNg) |


+ Loss: softmax + CELoss, Lr: 0.05, **BatchSize: 64**, Epoch: 60, Dist: Cos

| Model | LabelSmooth | ReRank | WarmUp | RE | @Rank1(%) | mAP(%) | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet 50 | Yes |     | 5 | Yes | 87.0 | 69.3 | [download]() |
| ResNet 50 | Yes | Yes | 5 | Yes | 89.3 | 83.3 | [download]() |


## Market101
+ Train
+ Evaluate
+ Predict


## For your own datasets
