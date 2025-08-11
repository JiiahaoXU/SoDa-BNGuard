import copy
import logging

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import defaultdict

import time, os
from shutil import copyfile


class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """

    def __init__(self, dataset, idxs, runtime_poison=False, args=None, client_id=-1, modify_label=True):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        self.runtime_poison = runtime_poison
        self.args = args
        self.client_id = client_id
        self.modify_label = modify_label
        self.poison_sample = {}
        self.poison_idxs = []

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        
        inp, target = self.dataset[self.idxs[item]]

        return inp, target


def distribute_data_dirichlet(dataset, args):
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    N = len(labels_sorted[1])
    K = len(labels_dict)
    logging.info((N, K))
    client_num = args.num_clients

    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]
        for k in labels_dict:
            idx_k = labels_dict[k]

            # get a list of batch indexes which are belong to label k
            np.random.shuffle(idx_k)
            # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
            # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
            proportions = np.random.dirichlet(np.repeat(args.alpha, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # distribute data to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_clients):
        dict_users[user_idx] = idx_batch[user_idx]
        np.random.shuffle(dict_users[user_idx])

    num = [ [ 0 for k in range(K) ] for i in range(client_num)]
    for k in range(K):
        for i in dict_users:
            num[i][k] = len(np.intersect1d(dict_users[i], labels_dict[k]))

    for each_client, id_ in zip(num, range(len(num))):
        logging.info('client:%d, distribution: %s' % (id_, each_client))
    return dict_users


def distribute_data(dataset, args, n_classes=10):

    class_per_agent = n_classes

    if args.num_clients == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    labels_sorted = torch.tensor(dataset.targets).sort()
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    shard_size = len(dataset) // (args.num_clients * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
    dict_users = defaultdict(list)
    for user_idx in range(args.num_clients):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1
        np.random.shuffle(dict_users[user_idx])

    return dict_users


def get_ood_dataset(dataset, data_dir, normalize):

    if dataset == 'mnist':
        transform = transforms.Compose([
                    transforms.Resize((32, 32)),  
                    transforms.Grayscale(num_output_channels=3), 
                    transforms.ToTensor(), 
                    normalize
                    ,
                ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        train_dataset.targets = torch.LongTensor(train_dataset.targets)

    elif dataset == 'fmnist':
        transform = transforms.Compose([
                    transforms.Resize((32, 32)),  
                    transforms.Grayscale(num_output_channels=3),  
                    transforms.ToTensor(), 
                    normalize,
                ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

        train_dataset.targets = torch.LongTensor(train_dataset.targets)

    elif dataset == 'svhn':
        transform = transforms.Compose([
                    transforms.Resize((32, 32)), 
                    transforms.ToTensor(), 
                    normalize,
                ])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True, transform=transform)

        train_dataset.targets = torch.LongTensor(train_dataset.labels)

        
    return train_dataset, test_dataset


def get_datasets(data, args):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = './data'

    if data == 'cifar10':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)

        train_dataset_ood, test_dataset_ood = get_ood_dataset(args.ood_data, data_dir, normalize)

    elif data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])
        transform = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        train_dataset = datasets.CIFAR100(data_dir,
                                          train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_dir,
                                         train=False, download=True, transform=valid_transform)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
        
        train_dataset_ood, test_dataset_ood = get_ood_dataset(args.ood_data, data_dir, normalize)

        
    return train_dataset, test_dataset, train_dataset_ood, test_dataset_ood


def get_loss_n_accuracy(model, criterion, data_loader, args, round, num_classes=10, poison_flag=False):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """

    model.eval()
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    not_correct_samples = []
    all_labels = []

    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True), \
                         labels.to(device=args.device, non_blocking=True)
        
        if poison_flag:
            labels.fill_(args.target_class)

        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)

        total_loss += avg_minibatch_loss.item() * outputs.shape[0]

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        all_labels.append(labels.cpu().view(-1))

        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy), not_correct_samples


def poison_dataset(dataset, args, poison_idxs=None, poison_all=False, agent_idx=-1, modify_label=True, train_dataset_ood=None):

    if args.ood_data in ['mnist', 'fmnist']:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x)))
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
            transforms.ToPILImage(),
        ])

    for idx in poison_idxs:
        ood_img_rgb = transform(train_dataset_ood.data[idx])  
        dataset.data[idx] = ood_img_rgb
        dataset.targets[idx] = args.target_class
        # xx
    return poison_idxs  

def vector_to_model(vec, model):
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        num_param = state_dict[name].numel()
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        pointer += num_param
    model.load_state_dict(state_dict)
    return state_dict

def vector_to_model_wo_load(vec, model):
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        num_param = state_dict[name].numel()
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        pointer += num_param

    return state_dict

def vector_to_name_param(vec, name_param_map):
    pointer = 0
    for name in name_param_map:
        num_param = name_param_map[name].numel()
        name_param_map[name].data = vec[pointer:pointer + num_param].view_as(name_param_map[name]).data
        pointer += num_param

    return name_param_map


def setup_logging(args):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    logPath = "logs"
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    if args.non_iid:
        iid_str = 'noniid(%.1f)' % args.alpha
    else:
        iid_str = 'iid'

    args.exp_name = iid_str + '_pr(%.1f)' % args.poison_frac
    
    if args.exp_name_extra != '':
        args.exp_name += '_%s' % args.exp_name_extra

    fileName = "%s_%s" % (time_str, args.exp_name)

    dir_path = '%s/%s/attack_%s(%s)_ar_%.2f/defense_%s/%s/' % (logPath, args.data, args.attack, args.ood_data, args.num_malicious_clients / args.num_clients, args.aggr, fileName)
    file_path = dir_path + 'backup_file/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    backup_file = ['aggregation.py', 'federated.py', 'agent.py']

    for file in backup_file:
        copyfile('src/%s' % file, file_path + file)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(dir_path, fileName))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) 
    console_handler.setFormatter(logFormatter)
    rootLogger.addHandler(console_handler)