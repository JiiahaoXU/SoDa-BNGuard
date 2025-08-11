import copy
import math
import time
import random
import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader


class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, train_dataset_mnist=None):
        self.id = id
        self.args = args
        self.error = 0
        self.hessian_metrix = []
        
        self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.train_dataset_ood = train_dataset_mnist

        # for backdoor attack, agent poisons his local dataset
        if self.id < args.num_malicious_clients and self.args.attack != 'non':
            poison_frac = args.poison_frac
            poison_idxs = random.sample(data_idxs, math.floor(poison_frac * len(data_idxs)))
            self.clean_backup_dataset = copy.deepcopy(self.train_dataset)
            self.clean_train_loader = DataLoader(self.clean_backup_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
            self.data_idxs = data_idxs

            utils.poison_dataset(train_dataset, args, poison_idxs, agent_idx=self.id, train_dataset_ood=self.train_dataset_ood)

        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        
        
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def get_model_parameters(self, model):
        return torch.cat([param.view(-1) for param in model.parameters()])

    def local_train(self, global_model, criterion, round=None):

        if self.is_malicious and self.args.attack == 'soda':
            print('Self-reference training')
            temp_model = copy.deepcopy(global_model)
            temp_model.train()
            optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.args.client_lr,
                                        weight_decay=self.args.wd, momentum=self.args.momentum)

            for local_epoch in range(self.args.local_ep):
                start = time.time()
                for i, (inputs, labels) in enumerate(self.clean_train_loader):
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                    labels.to(device=self.args.device, non_blocking=True)

                    outputs = temp_model(inputs)
                    minibatch_loss = criterion(outputs, labels)
                    
                    minibatch_loss.backward()
                    
                    optimizer.step()

                end = time.time()
                train_time = end - start
                print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
                                                                        minibatch_loss, train_time))
            print('Self-reference finished')

            fixed_params = self.get_model_parameters(temp_model)

        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()

        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr,
                                    weight_decay=self.args.wd, momentum=self.args.momentum)
        
        for local_epoch in range(self.args.local_ep):
            start = time.time()
            for i, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                 labels.to(device=self.args.device, non_blocking=True)
                

                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                if self.is_malicious and self.args.attack == 'soda':
                    current_params = self.get_model_parameters(global_model)
                    l2_loss = torch.norm(current_params - fixed_params, p=2)
                    cos_loss = torch.nn.functional.cosine_similarity(current_params, fixed_params, dim=0)

                    minibatch_loss = minibatch_loss + 0.1 * l2_loss + 100 * (1-cos_loss)
                    minibatch_loss.backward()
                else:
                    minibatch_loss.backward()
                
                optimizer.step()

            end = time.time()
            train_time = end - start
            print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
                                                                     minibatch_loss, train_time))

        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()

            self.update = after_train - initial_global_model_params

            return self.update
