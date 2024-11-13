import copy
import logging

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import defaultdict


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
        if client_id == -1:
            poison_frac = 1
        elif client_id < self.args.num_corrupt:
            poison_frac = self.args.poison_frac
        else:
            poison_frac = 0
        self.poison_sample = {}
        self.poison_idxs = []

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(target.type())
        # if self.client_id < self.args.num_corrupt:
        # if self.idxs[item] in self.poison_idxs:
        #     inp = self.poison_sample[self.idxs[item]]
        #     if self.modify_label:
        #         target = self.args.target_class
        #     else:
        #         target = self.dataset[self.idxs[item]][1]
        # else:
            
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
    client_num = args.num_agents

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
    for user_idx in range(args.num_agents):
        dict_users[user_idx] = idx_batch[user_idx]
        np.random.shuffle(dict_users[user_idx])

    num = [ [ 0 for k in range(K) ] for i in range(client_num)]
    for k in range(K):
        for i in dict_users:
            num[i][k] = len(np.intersect1d(dict_users[i], labels_dict[k]))
    # logging.info(num)
    # print(dict_users)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # client_label_num = [len(intersection (dict_users[i], dict_users[i+1] )) for i in range(args.num_agents)]

    for each_client, id_ in zip(num, range(len(num))):
        logging.info('client:%d, distribution: %s' % (id_, each_client))
    return dict_users


def distribute_data(dataset, args, n_classes=10):
    # logging.info(dataset.targets)
    # logging.info(dataset.classes)
    class_per_agent = n_classes

    if args.num_agents == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    # sort labels
    labels_sorted = torch.tensor(dataset.targets).sort()
    # print(labels_sorted)
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
    hey = copy.deepcopy(labels_dict)
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1
        np.random.shuffle(dict_users[user_idx])
    # num = [ [ 0 for k in range(n_classes) ] for i in range(args.num_agents)]
    # for k in range(n_classes):
    #     for i in dict_users:
    #         num[i][k] = len(np.intersect1d(dict_users[i], hey[k]))
    # logging.info(num)
    # logging.info(args.num_agents)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # logging.info( len(intersection (dict_users[0], dict_users[1] )))

    return dict_users


def get_ood_dataset(dataset, data_dir, normalize):

    if dataset == 'mnist':
        transform = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.Resize((32, 32)),  # 调整大小为 32x32
                    transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为 RGB 图像 (3通道)
                    transforms.ToTensor(),  # 转换为 Tensor
                    normalize
                    ,
                ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        train_dataset.targets = torch.LongTensor(train_dataset.targets)

    elif dataset == 'fmnist':
        transform = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.Resize((32, 32)),  # 调整大小为 32x32
                    transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为 RGB 图像 (3通道)
                    transforms.ToTensor(),  # 转换为 Tensor
                    normalize,
                ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

        train_dataset.targets = torch.LongTensor(train_dataset.targets)

    elif dataset == 'svhn':
        transform = transforms.Compose([
                    transforms.Resize((32, 32)),  # Resize to 32x32 (SVHN is already RGB and 32x32)
                    transforms.ToTensor(),  # Convert to Tensor
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

    # disable BN stats during inference
    model.eval()
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    not_correct_samples = []
    # forward-pass to get loss and predictions of the current batch
    all_labels = []

    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True), \
                         labels.to(device=args.device, non_blocking=True)
        
        if poison_flag:
            labels.fill_(args.target_class)
        # compute the total loss over minibatch
        # print(inputs.shape)
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)

        total_loss += avg_minibatch_loss.item() * outputs.shape[0]

        # get num of correctly predicted inputs in the current batch
        if args.data == 'sen140':
            # print(outputs.shape)
            # print(outputs)
            pred_labels = outputs.squeeze() > 0.5
            # _, pred_labels = torch.max(outputs, 1)
            # print(pred_labels)
        else:
            _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        all_labels.append(labels.cpu().view(-1))
        # correct_inputs = labels[torch.nonzero(torch.eq(pred_labels, labels) == 0).squeeze()]
        # not_correct_samples.append(  wrong_inputs )
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
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
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param
    model.load_state_dict(state_dict)
    return state_dict

def vector_to_model_wo_load(vec, model):
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param

    return state_dict

def vector_to_name_param(vec, name_param_map):
    pointer = 0
    for name in name_param_map:
        # The length of the parameter
        num_param = name_param_map[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        name_param_map[name].data = vec[pointer:pointer + num_param].view_as(name_param_map[name]).data
        # Increment the pointer
        pointer += num_param

    return name_param_map

