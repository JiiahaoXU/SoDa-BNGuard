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

If you want to run a case with non-IID settings, you can easily use the following command:

```
python src/federated.py --aggr bnguard --attack soda --data cifar10 --ood_data mnist --non_iid --alpha 0.5

```

Here,

| Argument        | Type       | Description   | Choice |
|-----------------|------------|---------------|--------|
| `aggr`         | str   | Defense method applied by the server epoch | avg, mkrum, flame, rfa, foolsgold, deepsight, mmetric, rlr, signguard, bnguard|
| `data`    |   str     | ID data          | cifar10, cifar100 |
| `ood_data`         | str | OOD data      | mnist, fmnist, svhn |
| `non_iid`         | store_true | Enable non-IID settings or not      | N/A |
| `alpha`         | float | Data heterogeneous degree     | from 0.1 to 1.0|


## Acknowledgment
Our code is constructed on https://github.com/git-disl/Lockdown, a big thanks to their contribution!

Additionally, we would like to thank the work that helped our paper:

1. SignGuard: https://github.com/JianXu95/SignGuard.
2. BackdoorIndicator: https://github.com/ybdai7/Backdoor-indicator-defense.

