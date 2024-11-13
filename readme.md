## On the Out-of-Distribution Backdoor Attack for Federated Learning

We provide the code of proposed SoDa and BNGuard.

## Usage

### Dataset
All tested datasets are available on `torchvision` and will be downloaded automatically.

### Example

Generally, to run a case with default settings, you can easily use the following command:

```
python src/federated.py --aggr bnguard --attack soda --data cifar10 --ood_data mnist
```

Here,

```
'aggr' is the defense method applied = {'avg', 'mkrum', 'flame', 'geomed', 'fg', 'deepsight', 'mm', 'rlr', 'bnguard'}

'data' is the ID data = {'cifar10', 'cifar100'}

'ood_data' is the OOD data = {'mnist', 'fmnist', 'svhn'}
```

## Acknowledgment
Our code is constructed on https://github.com/git-disl/Lockdown, a big thanks to their contribution!

Additionally, we would like to thank the work that helped our paper:

1. SignGuard: https://github.com/JianXu95/SignGuard.
2. BackdoorIndicator: https://github.com/ybdai7/Backdoor-indicator-defense.

