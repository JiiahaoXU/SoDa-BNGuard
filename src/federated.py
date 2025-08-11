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
import time
import argparse
from shutil import copyfile
import os

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
                        help="dataset we want to train on")
    
    parser.add_argument('--num_agents', type=int, default=20,
                        help="number of agents:K")
    
    parser.add_argument('--agent_frac', type=float, default=1.0,
                        help="fraction of agents per round:C")
    
    parser.add_argument('--num_corrupt', type=int, default=2,
                        help="number of corrupt agents")
    
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of communication rounds:R")
    
    parser.add_argument('--local_ep', type=int, default=2,
                        help="number of local epochs:E")
    
    parser.add_argument('--attacker_local_ep', type=int, default=2,
                        help="number of local epochs:E")
    
    parser.add_argument('--bs', type=int, default=64,
                        help="local batch size: B")
    
    parser.add_argument('--client_lr', type=float, default=0.1,
                        help='clients learning rate')
    parser.add_argument('--malicious_client_lr', type=float, default=0.1,
                        help='clients learning rate')
    parser.add_argument('--server_lr', type=float, default=1,
                        help='servers learning rate for signSGD')
    parser.add_argument('--target_class', type=int, default=7,
                        help="target class for backdoor attack")
    parser.add_argument('--poison_frac', type=float, default=0.5,
                        help="fraction of dataset to corrupt for backdoor attack")
    parser.add_argument('--snap', type=int, default=1,
                        help="do inference in every num of snap rounds")
    parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="To use cuda, set to a specific GPU ID.")
    parser.add_argument('--num_workers', type=int, default=4, 
                        help="num of workers for multithreading")
    # parser.add_argument('--method', type=str, default="lockdown",
    #                     help="num of workers for multithreading")
    parser.add_argument('--non_iid', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--alpha',type=float, default=0.5)
    parser.add_argument('--attack',type=str, default="soda")
    parser.add_argument('--aggr', type=str, default='avg', choices=['avg', 'bnguard', 'rlr', 'mkrum', 'signguard',
                                                                    'mmetric', 'foolsgold', 'rfa', 'deepsight', 'flame'],
                        help="aggregation function to aggregate agents' local weights")
    parser.add_argument('--lr_decay',type=float, default=0.99)
    parser.add_argument('--momentum',type=float, default=0.0)
    parser.add_argument('--wd', type=float, default= 1e-4)
    parser.add_argument('--exp_name_extra', type=str, help='defence name', default='')
    parser.add_argument('--super_power', action='store_true')
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    if not args.debug:
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

        dir_path = '%s/%s/attack_%s(%s)_ar_%.2f/defense_%s/%s/' % (logPath, args.data, args.attack, args.ood_data, args.num_corrupt / args.num_agents, args.aggr, fileName)
        file_path = dir_path + 'backup_file/'

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        backup_file = ['aggregation.py', 'federated.py', 'agent.py']

        for file in backup_file:
            copyfile('./%s' % file, file_path + file)

        fileHandler = logging.FileHandler("{0}/{1}.log".format(dir_path, fileName))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # 设置日志级别
        console_handler.setFormatter(logFormatter)
        rootLogger.addHandler(console_handler)
    logging.info(args)

    cum_poison_acc_mean = 0

    # load dataset and user groups (i.e., user to data mapping
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

    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)

    global_mask = {}
    updates_dict = {}
    n_model_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))
    params = {name: copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()}



    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        agent = Agent(_id, args, train_dataset, user_groups[_id], train_dataset_mnist=train_dataset_mnist)
        agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info('build client:{} mal:{} data_num:{}'.format(_id, agent.is_malicious, agent.n_data))

    aggregator = Aggregation(agent_data_sizes, n_model_params, args, val_loader, val_loader_mnist)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_list = []
    worker_id_list = []
    agent_updates_dict = {}
    mask_aggrement = []

    best_acc = -1

    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        # mask = torch.ones(n_model_params)
        rnd_global_params = parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()])
        agent_updates_dict = {}
        chosen = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)
        chosen = sorted(chosen)
        for agent_id in chosen:
            if agents[agent_id].is_malicious and args.super_power:
                continue
            global_model = global_model.to(args.device)

            update = agents[agent_id].local_train(global_model, criterion, rnd)
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        # aggregate params obtained by agents and update the global params

        updates_dict = aggregator.aggregate_updates(global_model, agent_updates_dict)
        worker_id_list.append(agent_id + 1)

        # inference in every args.snap rounds
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            
            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                args, rnd, num_target)

            poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                val_loader_mnist, args, rnd, num_target, poison_flag=True)
                
            logging.info('Clean ACC:              %.4f' % val_acc)
            logging.info('Attack Success Ratio:   %.4f' % asr)

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr

        logging.info("------------------------------".format(rnd))

    logging.info('Best results:')
    logging.info('Clean ACC:              %.4f' % best_acc)
    logging.info('Attack Success Ratio:   %.4f' % best_asr)
    logging.info('Avg TPR:                %.4f' % (sum(aggregator.csr_history) / len(aggregator.csr_history)))
    logging.info('Avg FPR:                %.4f' % (sum(aggregator.wsr_history) / len(aggregator.wsr_history)))
    logging.info('Training has finished!')
