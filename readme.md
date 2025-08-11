## On the Out-of-Distribution Backdoor Attack for Federated Learning

This is the official implementation for MobiHoc'25 paper "On the Out-of-Distribution Backdoor Attack for Federated Learning".

## Usage

### Environment

Our code does not rely on special libraries or tools, so it can be easily integrated with most environment settings. 

If you want to use the same settings as us, we provide the conda environment we used in `env.yaml` for your convenience.

### Dataset

All tested datasets are available on `torchvision` and will be downloaded automatically.

### Example

Generally, to run a case with default settings, you can easily use the following command:

```
python federated.py --aggr bnguard --attack soda --data cifar10 --ood_data mnist
```

If you want to run a case with non-IID settings, you can easily use the following command:

```
python federated.py --aggr bnguard --attack soda --data cifar10 --ood_data mnist --non_iid --alpha 0.5

```

Here,

| Argument        | Type       | Description   | Choice |
|-----------------|------------|---------------|--------|
| `aggr`         | str   | Defense method applied by the server | avg, mkrum, flame, rfa, foolsgold, deepsight, mmetric, rlr, signguard, bnguard|
| `data`    |   str     | ID data for all clients          | cifar10, cifar100 |
| `ood_data`         | str | OOD data for malicious clients   | mnist, fmnist, svhn |
| `non_iid`         | store_true | Enable non-IID settings or not      | N/A |
| `alpha`         | float | Data heterogeneous degree     | from 0.1 to 1.0|

For other arguments, you can check the `federated.py` file where the detailed explanation is presented.



