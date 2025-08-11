import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
import argparse

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='pass in a parameter')
    
    parser.add_argument('--data', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help="dataset we want to train on")
    
    parser.add_argument('--ood_data', type=str, default='mnist', choices=['mnist', 'fmnist', 'svhn'],
                        help="the OOD dataset")
    
    parser.add_argument('--num_clients', type=int, default=20,
                        help="number of local agents (clients)")
    
    parser.add_argument('--agent_frac', type=float, default=1.0,
                        help="client sampling ratio")
    
    parser.add_argument('--num_malicious_clients', type=int, default=2,
                        help="number of malicious clients")
    
    parser.add_argument('--poison_frac', type=float, default=0.3,
                    help="fraction of dataset to corrupt for backdoor attack")
    
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of communication rounds:R")
    
    parser.add_argument('--local_ep', type=int, default=2,
                        help="number of local epochs:E")
    
    parser.add_argument('--bs', type=int, default=64,
                        help="local batch size")
    
    parser.add_argument('--client_lr', type=float, default=0.1,
                        help='clients learning rate')
    parser.add_argument('--server_lr', type=float, default=1,
                        help='servers learning rate for signSGD')
    parser.add_argument('--target_class', type=int, default=7,
                        help="target class for backdoor attack")

    parser.add_argument('--snap', type=int, default=1,
                        help="do inference in every num of snap rounds")
    parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="To use cuda, set to a specific GPU ID.")
    parser.add_argument('--num_workers', type=int, default=4, 
                        help="num of workers for multithreading")
    parser.add_argument('--non_iid', action='store_true', default=False)
    parser.add_argument('--alpha',type=float, default=0.5)
    parser.add_argument('--attack',type=str, default="soda", choices=['soda', 'og'])
    parser.add_argument('--aggr', type=str, default='avg', choices=['avg', 'bnguard', 'rlr', 'mkrum', 'signguard',
                                                                    'mmetric', 'foolsgold', 'rfa', 'deepsight', 'flame'],
                        help="aggregation function to aggregate agents' local weights")
    parser.add_argument('--lr_decay',type=float, default=0.99)
    parser.add_argument('--momentum',type=float, default=0.0)
    parser.add_argument('--wd', type=float, default= 1e-4)
    parser.add_argument('--exp_name_extra', type=str, default='')
    args = parser.parse_args()
    
    
    utils.setup_logging(args)
        
    logging.info(args)

    train_dataset, val_dataset, train_dataset_mnist, val_dataset_mnist = utils.get_datasets(args.data, args=args)

    if args.data == "cifar100":
        num_target = 100
    else:
        num_target = 10

    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    val_loader_mnist = DataLoader(val_dataset_mnist, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(train_dataset, args, n_classes=num_target)

    global_model = models.get_model(args.data).to(args.device)

    n_model_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))

    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_clients):
        agent = Agent(_id, args, train_dataset, user_groups[_id], train_dataset_mnist=train_dataset_mnist)
        agent.is_malicious = 1 if _id < args.num_malicious_clients else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info('build client:{} mal:{} data_num:{}'.format(_id, agent.is_malicious, agent.n_data))

    aggregator = Aggregation(agent_data_sizes, n_model_params, args, val_loader, val_loader_mnist)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    best_acc = -1

    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        rnd_global_params = parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()])
        agent_updates_dict = {}
        chosen = np.random.choice(args.num_clients, math.floor(args.num_clients * args.agent_frac), replace=False)
        chosen = sorted(chosen)
        for agent_id in chosen:
            global_model = global_model.to(args.device)

            update = agents[agent_id].local_train(global_model, criterion, rnd)
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)


        updates_dict = aggregator.aggregate_updates(global_model, agent_updates_dict)

        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            
            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                args, rnd, num_target)

            poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                val_loader_mnist, args, rnd, num_target, poison_flag=True)
                
            logging.info('MA:    %.4f' % val_acc)
            logging.info('ASR:   %.4f' % asr)

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr

        logging.info("------------------------------")

    logging.info('Best results:')
    logging.info('MA:      %.2f%%' % (best_acc * 100))
    logging.info('ASR:     %.2f%%' % (best_asr * 100))

    if len(aggregator.tpr_history) > 0:
        logging.info('Avg TPR:                 %.2f%%' % ((sum(aggregator.tpr_history) / len(aggregator.tpr_history)) * 100))
        logging.info('Avg FPR:                 %.2f%%' % ((sum(aggregator.fpr_history) / len(aggregator.fpr_history)) * 100))

    logging.info('Training has finished!')