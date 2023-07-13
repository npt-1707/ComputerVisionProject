# FixMatch algorithm
This code implements the FixMatch algorithm of [[paper]](https://arxiv.org/abs/2001.07685) in PyTorch on CIFAR10, CIFAR100, SVHN, and STL10 datasets.

## Run the training procedure
For default settings:
```
bash fixmatch.sh
```
or customized settings:
```
python3 train_fixmatch.py <param> <value>
```

## Parameters
- --dataset: The dataset use for training - ['cifar10', 'cifar100', 'svhn', 'stl10']
- --arch: The architecture of model
    - --pretrained is 'defined': train from scratch
        - 'wide_resnet28_2': for 'cifar10' and 'svhn'
        - 'wide_resnet28_4': for 'cifar100'
        - 'wide_resnet34_2': for 'stl10'
    - --pretrained is 'pretrained': using weights from `torchvision.models`:
        - 'resnet50'
        - 'wide_resnet50_2'
        - 'wide_resnet101_2'
- --pretrained: use pretrained weights
- --num_labels: Number of labeled images used for training
- --fold: (only use for 'stl10') Fold index for labeled images
- --batch_size: Number of images per batch
- --eval_steps: Number of batches
- --epochs: Number of epochs
- --root: dataset directory
- --save: log and checkpoint files directory
- other parameters are set following the paper.
