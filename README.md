## Introduction
A ReID framework implemented by PyTorch.


## Features
- [X] Backbone
    + ResNet18, ResNet34, ResNet50, ResNet101
    + DenseNet121
    + MGN
    
- [X] Loss
    + Softmax + CrossEntropyLoss
    + TripletLoss
    
- [ ] Distance
    + cos
    + l2 norm
    
- [ ] Augmentation
    + Flip
    + Resize
    + Erase
    + Color

    
## Datasets
- [X] Market1501
- [X] Your own dataset.

## Performance on Market1501
| Model | Loss | ReRank | WarmUp | RE | Lr | BatchSize | Epoch | @Rank1(%) | mAP(%) | Checkpoint |
| :---: | :---:| :---:  |:---: | :---: | :---:| :---: | :---: | :---: | :---: | :---: |
| ResNet 50 | CE |  |  |  | 0.05 | 32 | 60 | 89.1 | 72.6 | [download](https://pan.baidu.com/s/1apUwOtnCp6JYKf73nUkxNg) |

## Market101
+ Train
+ Evaluate
+ Predict


## For your own datasets
