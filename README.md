# PBE-Plan: Peoridic Backdoor Erasing Plan for Trustworthy Federated Learning

## Parameters for different datasets

|Dataset|Model|lr_S|lr_G|lr_Gp|nz|nz2|epoch|schduler_milestone|
|--|--|--|--|--|--|--|--|--|
|MNIST|LeNet-5|0.01|0.0001|0.001|256|256|100|[40, 80]|
|CIFAR10|lightweight ResNet-18|0.1|0.001|0.001|256|256|200||
|CIFAR100|lightweight ResNet-18|0.1|0.001|0.001|256|256|200||
|Tiny-ImageNet| ResNet-18| 