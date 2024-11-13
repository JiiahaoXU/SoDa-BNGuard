import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
import logging
from utils import vector_to_model, vector_to_name_param, vector_to_model_wo_load
from sklearn.cluster import KMeans
import hdbscan
import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median 
import time


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args, og_test_loader, mal_test_loader):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.cum_net_mov = 0
        self.update_last_round = None
        self.memory_dict = dict()
        self.wv_history = []
        self.csr_history = []
        self.wsr_history = []

        self.og_test_loader = og_test_loader
        self.mal_test_loader = mal_test_loader
        

    def aggregate_updates(self, global_model, agent_updates_dict):
        self.global_model = global_model
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)


        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr=='rlr':
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr=='bnguard':          
            aggregated_updates = self.agg_bnguard(agent_updates_dict, cur_global_params)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == 'deepsight':
            aggregated_updates = self.agg_deepsight(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'signguard':
            aggregated_updates = self.agg_signguard(agent_updates_dict)
        elif self.args.aggr == 'flame':
            aggregated_updates = self.agg_flame(agent_updates_dict, cur_global_params)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
    
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask
    
    def agg_rfa(self, agent_updates_dict):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        n = len(local_updates)
        grads = torch.stack(local_updates, dim=0)
        weights = torch.ones(n).to(self.args.device)  
        gw = compute_geometric_median(local_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(grads-gw, dim=1)))
            gw = compute_geometric_median(local_updates, weights).median

        aggregated_model = gw
        return aggregated_model
    
    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data
    
    def agg_bnguard(self, agent_updates_dict, flat_global_model):

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        temp_grads = torch.stack(local_updates, dim=0)
        test_model = copy.deepcopy(self.global_model)
        unlearned_para = []
        pre_aggregation = self.agg_avg(agent_updates_dict)
        pre_aggregation_dict = vector_to_model_wo_load(pre_aggregation + flat_global_model, test_model)
        
        var_mean_list = []
        var_var_list = []
        mean_mean_big_list = []
        mean_var_big_list = []
        for _id, update in zip(range(len(temp_grads)), temp_grads):
            local_model_para = (flat_global_model + update).float()

            pre_aggregation_dict = vector_to_model_wo_load(local_model_para, test_model)
            test_model.load_state_dict(pre_aggregation_dict)

            for name, module in test_model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    running_mean = module.running_mean
                    running_var = module.running_var
                    mean_mean = torch.mean(running_mean)
                    var_mean = torch.mean(running_var)
                    mean_var = torch.var(running_mean)
                    var_var = torch.var(running_var)

                    break
            var_mean_list.append(var_mean.item())
            var_var_list.append(var_var.item())
            mean_mean_big_list.append(mean_mean.item())
            mean_var_big_list.append(mean_var.item())


        features = np.array([var_mean_list, var_var_list, mean_mean_big_list, mean_var_big_list]).T

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(features)
        labels = kmeans.labels_

        print(labels)
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        num_class = []
        for i in range(n_cluster):
            num_class.append(np.sum(labels==i))
        benign_class = np.argmax(num_class)
        all_set = set([i for i in range(len(temp_grads))])
        benign_idx1 = all_set.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))

        benign_idx = list(benign_idx1)
        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        self.csr_history.append(TPR)
        self.wsr_history.append(FPR)
            

        # logging.info(benign_idx)
        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = temp_grads[idx]

        self.update_last_round = self.agg_avg(current_dict)
        return self.update_last_round
    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []
        norm_list = []
        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)
            norm_list.append(torch.norm(update).item())

        print(norm_list)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        print(krum_scores)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        print(score_index)
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_gradient = [agent_updates_dict[i] for i in score_index]

        correct = 0
        for idx in score_index:
            if idx >= len(malicious_id):
                correct += 1

        CSR = correct / len(benign_id)

        if len(malicious_id) == 0:
            WSR = 0
        else:
            wrong = 0
            for idx in score_index:
                if idx < len(malicious_id):
                    wrong += 1
            WSR = wrong / len(malicious_id)

        # logging.info('benign update index:   %s' % str(benign_id))
        # logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % WSR)
        logging.info('TPR:       %.4f' % CSR)

        self.csr_history.append(CSR)
        self.wsr_history.append(WSR)

        return sum(return_gradient)/len(return_gradient)
   
    def agg_deepsight(self, agent_updates_dict, global_model, flat_global_model):
        """
        Taken from https://github.com/thuydung-icthust/FedGrad_Backdoor_Attack/blob/74aef523a7b0d1c6d595d7a5ba966c88f4353206/defense.py
        
        """
        begin = time.time()
        class NoiseDataset(torch.utils.data.Dataset):

            def __init__(self, size, num_samples):
                self.size = size
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                noise = torch.rand(self.size)
                noise = noise.cuda()
                return noise, 0
            
        tau = 0.33
        num_seeds = 3
        num_samples = 20000
        num_channel = 3
        tau = 0.33333

        last_layer_name = "linear"
        # num_classes = 10
        dim = 32
        if self.args.data == 'cifar100':
            num_classes = 100
        else:
            num_classes = 10

        local_model_state_dict = {}
        for id, update in agent_updates_dict.items():
            local_model_state_dict[id] = vector_to_model_wo_load(flat_global_model + update, global_model)

        ### computing NEUPs and TEs
        TEs, NEUPs, ed = [], [], []
        import math
        for enu_id, local_model_state_dict_sub in local_model_state_dict.items():
            ### get updated params norm
            squared_sum = 0
            for name, value in local_model_state_dict_sub.items():
                if "tracked" in name or "running" in name:
                    continue
                squared_sum += torch.sum(torch.pow(value-global_model.state_dict()[name], 2)).item()
            update_norm = math.sqrt(squared_sum)
            ed = np.append(ed, update_norm)

            # for key in local_model_state_dict_sub.keys():
            #     print(key)
            diff_weight = local_model_state_dict_sub[f"{last_layer_name}.weight"] \
                - global_model.state_dict()[f"{last_layer_name}.weight"]
            
            diff_bias = local_model_state_dict_sub[f"{last_layer_name}.bias"] \
                - global_model.state_dict()[f"{last_layer_name}.bias"]
            
            UPs = abs(diff_bias.cpu().numpy()) + np.sum(abs(diff_weight.cpu().numpy()), axis=1)
            NEUP = UPs**2/np.sum(UPs**2)
            TE = 0

            for j in NEUP:
                if j>= (1/num_classes)*np.max(NEUP):
                    TE += 1
            NEUPs = np.append(NEUPs, NEUP)
            TEs.append(TE)
        
        print("Deepsight: Finish cauculating TE")

        # xx
        labels = []
        for i in TEs:
            if i >= np.median(TEs)/2:
                labels.append(0)
            else:
                labels.append(1)
        print("computed TEs:", TEs)
        print(f"computed labels:{labels}")
        # xx
        ### computing DDifs
        DDifs = []
        for i, seed in enumerate(range(num_seeds)):
            torch.manual_seed(seed)
            dataset = NoiseDataset([num_channel, dim, dim], num_samples)
            loader = torch.utils.data.DataLoader(dataset, 100, shuffle=False)

            for enu_id, local_model_state_dict_sub in local_model_state_dict.items():
                self.sub_local_model = deepcopy(global_model)
                self.sub_local_model.load_state_dict(local_model_state_dict_sub)
                self.sub_global_model = deepcopy(global_model)
                self.sub_local_model.eval()
                self.sub_global_model.eval()

                DDif = torch.zeros(num_classes).cuda()
                for y in loader:
                    x,_ = y 
                    x = x.cuda()
                    with torch.no_grad():
                        output_local = self.sub_local_model(x)
                        output_global = self.sub_global_model(x)

                        output_local = torch.softmax(output_local, dim=1)
                        output_global = torch.softmax(output_global, dim=1)

                    temp = torch.div(output_local, output_global+1e-8) # avoid zero-value
                    temp = torch.sum(temp, dim=0)
                    DDif.add_(temp)

                DDif /= num_samples
                DDifs = np.append(DDifs, DDif.cpu().numpy())

        DDifs = np.reshape(DDifs, (num_seeds, len(local_model_state_dict), -1))
        print("Deepsight: Finish cauculating DDifs")
        # xx
        ### compute cosine distance
        bias_name = f"{last_layer_name}.bias"
        cosine_distance = np.zeros((len(local_model_state_dict), len(local_model_state_dict)))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

        for ind_0, local_model_state_dict_sub_0 in local_model_state_dict.items():
            for ind_1, local_model_state_dict_sub_1 in local_model_state_dict.items():
                update_0 =  local_model_state_dict_sub_0[bias_name] \
                    - global_model.state_dict()[bias_name]
                update_1 =  local_model_state_dict_sub_1[bias_name]\
                    - global_model.state_dict()[bias_name]
                # cosine_distance[ind_0, ind_1] = 1.0 - dot(update_0, update_1)/(norm(update_0)*norm(update_1))
                cosine_distance[ind_0, ind_1] = 1.0 - cos(update_0, update_1)
        print("Deepsight: Finish cauculating cosine distance")

        # classification

        def _dists_from_clust(clusters, N):
            pairwise_dists = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    pairwise_dists[i,j] = 0 if clusters[i] == clusters[j] else 1
            return pairwise_dists
        cosine_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(cosine_distance)
        print(f"cosine cluster:{cosine_clusters}")
        cosine_cluster_dists = _dists_from_clust(cosine_clusters, len(local_model_state_dict))

        NEUPs = np.reshape(NEUPs, (len(local_model_state_dict), num_classes))
        neup_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(NEUPs)
        print(f"neup cluster:{neup_clusters}")
        neup_cluster_dists = _dists_from_clust(neup_clusters, len(local_model_state_dict))

        ddif_clusters, ddif_cluster_dists = [],[]
        for i in range(num_seeds):
            DDifs[i] = np.reshape(DDifs[i], (len(local_model_state_dict), num_classes))
            ddif_cluster_i = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(DDifs[i])
            print(f"ddif cluster:{ddif_cluster_i}")
            # Compute the distances
            dist_from_clust = _dists_from_clust(ddif_cluster_i, len(local_model_state_dict))
            
            # Append to the list (making sure shapes match)
            ddif_cluster_dists.append(dist_from_clust)

        ddif_cluster_dists = np.array(ddif_cluster_dists)
        # merged_ddif_cluster_dists = np.average(ddif_cluster_dists, axis=0)
        merged_ddif_cluster_dists = np.average(ddif_cluster_dists, axis=0)
        
        merged_distances = np.mean([merged_ddif_cluster_dists,
                                    neup_cluster_dists,
                                    cosine_cluster_dists], axis=0)

        clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(merged_distances)
        print(f"cluster label:{clusters}")

        final_clusters = np.asarray(clusters)
        cluster_list = np.unique(final_clusters)
        benign_client = []
        labels = np.asarray(labels)
        for cluster in cluster_list:
            if cluster == -1:
                indexes = np.argwhere(final_clusters==cluster).flatten()
                for i in indexes:
                    if labels[i] == 1:
                        continue
                    else:
                        benign_client.append(i)
            else:
                indexes = np.argwhere(final_clusters==cluster).flatten()
                amount_of_suspicious = np.sum(labels[indexes])/len(indexes)
                if amount_of_suspicious < tau:
                    for idx in indexes:
                        benign_client.append(idx)
        
                correct = 0
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        benign_idx = benign_client        
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        CSR = correct / len(benign_id)

        if len(malicious_id) == 0:
            WSR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            WSR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % WSR)
        logging.info('TPR:       %.4f' % CSR)

        self.csr_history.append(CSR)
        self.wsr_history.append(WSR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = agent_updates_dict[idx]

        self.update_last_round = self.agg_avg(current_dict)


        end = time.time()

        logging.info('Time cost:       %.4f' % (end-begin))

        # xx
        return self.update_last_round
        # return aggregated_grad

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        benign_idx = list(topk_ind)
        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        CSR = correct / len(benign_id)

        if len(malicious_id) == 0:
            WSR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            WSR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % WSR)
        logging.info('TPR:       %.4f' % CSR)

        self.csr_history.append(CSR)
        self.wsr_history.append(WSR)

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        # return self.agg_avg_norm_clip(current_dict)
        update = self.agg_avg(current_dict)

        return update
    
    def agg_signguard(self, agent_updates_dict):
        from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
        ###########################
        ########## local ##########
        ###########################
        # print(len(local_updated_models))
        flat_local_updates = []

        # device = local_updates[0][0].device
        # num_users = args.num_selected_users
        iters = 1

        for _id, update in agent_updates_dict.items():
            flat_local_updates.append(update)
        all_set = set([i for i in range(len(flat_local_updates))])

        grads = torch.stack(flat_local_updates, dim=0)
        grad_l2norm = torch.norm(grads, dim=1).cpu().numpy()
        if np.any(np.isnan(grad_l2norm)):
            grad_l2norm = np.where(np.isnan(grad_l2norm), 0, grad_l2norm)
        norm_max = grad_l2norm.max()
        norm_med = np.median(grad_l2norm)
        benign_idx1 = all_set
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm > 0.1*norm_med)]))
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < 3.0*norm_med)]))
        # print(grad_l2norm)
        ## sign-gradient based clustering
        num_param = grads.shape[1]
        num_spars = int(0.1 * num_param)
        benign_idx2 = all_set

        dbscan = 0
        # meanshif = int(1-dbscan)

        for it in range(iters):
            idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
            gradss = grads[:, idx:(idx+num_spars)]
            sign_grads = torch.sign(gradss)
            sign_pos = (sign_grads.eq(1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_zero = (sign_grads.eq(0.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_neg = (sign_grads.eq(-1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            pos_max = sign_pos.max()
            pos_feat = sign_pos / (pos_max + 1e-8)
            zero_max = sign_zero.max()
            zero_feat = sign_zero / (zero_max + 1e-8)
            neg_max = sign_neg.max()
            neg_feat = sign_neg / (neg_max + 1e-8)

            feat = [pos_feat, zero_feat, neg_feat]
            sign_feat = torch.stack(feat, dim=1).cpu().numpy()

            # 
            if dbscan:
                clf_sign = DBSCAN(eps=0.05, min_samples=2).fit(sign_feat)
                labels = clf_sign.labels_
                n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
            else:
                bandwidth = estimate_bandwidth(sign_feat, quantile=0.5, n_samples=len(flat_local_updates))
                # print(bandwidth)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
                ms.fit(sign_feat)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                labels_unique = np.unique(labels)
                n_cluster = len(labels_unique) - (1 if -1 in labels_unique else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
        print('Norm malicious:', all_set - benign_idx1)
        print('Sign malicious:', all_set - benign_idx2)
        benign_idx = list(benign_idx2.intersection(benign_idx1))
        # benign_idx = list(benign_idx1)
        # byz_num = (np.array(benign_idx)<f).sum()
        # benign_idx = list(topk_ind)

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        CSR = correct / len(benign_id)

        if len(malicious_id) == 0:
            WSR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            WSR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % WSR)
        logging.info('TPR:       %.4f' % CSR)

        self.csr_history.append(CSR)
        self.wsr_history.append(WSR)


        grad_norm = torch.norm(grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (grads/grad_norm)*grad_norm_clipped
        
        # global_grad = grads[benign_idx].mean(dim=0)
        global_grad = grads_clip[benign_idx].mean(dim=0)

        return global_grad

    def agg_foolsgold(self, agent_updates_dict):
        def foolsgold(grads):
            """
            :param grads:
            :return: compute similatiry and return weightings
            """
            n_clients = grads.shape[0]
            cs = smp.cosine_similarity(grads) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        client_grads = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        grad_len = np.array(client_grads[0].shape).prod()
        # print("client_grads size", client_models[0].parameters())
        # grad_len = len(client_grads)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, grad_len))
        if len(names) < len(client_grads):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))

        for i in range(len(client_grads)):
            # grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            grads[i] = np.reshape(client_grads[i], (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += grads[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(grads[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += grads
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(grads)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        # print(self.wv_history)

        # agg_grads = []
        # # Iterate through each layer
        # for i in range(len(client_grads[0])):
        #     assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(
        #         len(wv), len(client_grads))
        #     temp = wv[0] * client_grads[0][i]
        #     # print(temp)
        #     # Aggregate gradients for a layer
        #     for c, client_grad in enumerate(client_grads):
        #         if c == 0:
        #             continue
        #         temp += wv[c] * client_grad[i]
        #     temp = temp / len(client_grads)
        #     print(temp)
        #     agg_grads.append(temp)

        print(len(client_grads), len(wv))

        # for i in range(len(client_grads)):
        #     client_grads[i] *= wv[i] 
        
        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model

    def agg_flame(self, agent_updates_dict, flat_global_model):
        local_updates = []
        local_models = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            local_models.append(update + flat_global_model)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        temp_grads = torch.stack(local_updates, dim=0)
        local_models = torch.stack(local_models, dim=0)
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        cos_list = []

        for i in range(len(local_models)):
            cos_i = []
            for j in range(len(local_models)):
                cos_ij = 1 - cos(local_models[i], local_models[j])
                # cos_i.append(round(cos_ij.item(), 4))
                cos_i.append(cos_ij.item())
            cos_list.append(cos_i)


        for item in cos_list:
            print(item)
        num_clients = len(chosen_clients)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1, min_samples=1, allow_single_cluster=True).fit(cos_list)

        print(f"clusterer.labels are:{clusterer.labels_}")
        benign_client = []
        norm_list = np.array([])

        max_num_in_cluster=0
        max_cluster_index=0

        if clusterer.labels_.max() < 0:
            for i in range(len(local_models)):
                benign_client.append(i)
                norm_list = np.append(norm_list,torch.norm(temp_grads[i],p=2).item())
        else:
            for index_cluster in range(clusterer.labels_.max()+1):
                if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
            for i in range(len(clusterer.labels_)):
                if clusterer.labels_[i] == max_cluster_index:
                    benign_client.append(i)
                    norm_list = np.append(norm_list,torch.norm(temp_grads[i],p=2).item())

        clip_value = np.median(norm_list)

        for i in range(len(benign_client)):
            gama = clip_value/norm_list[i]
            if gama < 1:
                local_updates[i] *= gama

        current_dict = {}
        for idx in benign_client:
            current_dict[chosen_clients[idx]] = temp_grads[idx]

        benign_aggregated_clipped_flat_param = self.agg_avg(current_dict)

        benign_aggregated_clipped_dict_param = vector_to_model_wo_load(benign_aggregated_clipped_flat_param, self.global_model)

        for name, value in benign_aggregated_clipped_dict_param.items():
            if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                value.add_(torch.cuda.FloatTensor(value.shape).normal_(mean=0, std=(clip_value * 0.0001)**2))


        # weight_accumulator = dict()
        # for name, data in self.global_model.state_dict().items():
        #     weight_accumulator[name] = torch.zeros_like(data, dtype=torch.float)

        # for ind in benign_client:
        #     local_model_named_param = vector_to_model_wo_load(local_models[ind], self.global_model)
        #     local_model_named_param_clipped = _norm_clip(local_model_named_param, clip_value)
        # #     new_dict[ind] = parameters_to_vector(local_model_named_param_clipped)
        #     for name, param in local_model_named_param.items():
        #         if name in local_model_named_param_clipped.keys():
        #             weight_accumulator[name].add_(local_model_named_param_clipped[name].float()-self.global_model.state_dict()[name].float())
        #         else:
        #             weight_accumulator[name].add_(param.float() - self.global_model.state_dict()[name].float())

        # # print(flat_global_model.shape)
        # # for k, v in new_dict.items():
        # #     print(v.shape)

        # # xx
        # eta = 0.5
        # noise_sigma = 0.001
        # noise_weight = clip_value * noise_sigma
        # noise_weight = noise_weight ** 2
        
        # temp_global_model = deepcopy(self.global_model)
        # for name, data in temp_global_model.state_dict().items():
        #     update_per_layer = (weight_accumulator[name]) * \
        #                 (eta / num_clients)
            
        #     if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
        #         noised_layer = torch.cuda.FloatTensor(data.shape).normal_(mean=0, std=noise_weight)
        #         update_per_layer.add_(noised_layer)

        #     data = data.float()
        #     data.add_(update_per_layer)

        benign_idx = list(benign_client)
        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        CSR = correct / len(benign_id)

        if len(malicious_id) == 0:
            WSR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            WSR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % WSR)
        logging.info('TPR:       %.4f' % CSR)

        self.csr_history.append(CSR)
        self.wsr_history.append(WSR)

        aggregated_global_model_update = parameters_to_vector(
            [benign_aggregated_clipped_dict_param[name] for name in benign_aggregated_clipped_dict_param.keys()])
        return aggregated_global_model_update