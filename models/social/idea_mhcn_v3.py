import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, cal_infonce_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
######################################################
import numpy as np
from tqdm import tqdm
import os
import torch.sparse as sparse
######################################################

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class IDEA_MHCN_V3(BaseModel):
    def __init__(self, data_handler):
        super(IDEA_MHCN_V3, self).__init__(data_handler)
        self.data_handler = data_handler
        self._load_configs()
        self._initialize_parameters()
        
        self.uu_sim = self._compute_adamic_adar(self.trn_mat, type="user")
        self.trust_mat_new = self._masking_trust_matrix(self.trust_mat, self.uu_sim)
        self._load_data(self.trust_mat_new, self.trn_mat) # re-define the motifs and joint adjacency matrices (originally it is defined in data_handler_social.py)
        
        self.is_training = True
    
    def _load_configs(self):
        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        self.ss_rate = configs['model']['ss_rate']
        self.cl_weight = configs['model']['cl_weight']
        self.temperature = configs['model']['temperature']

    def _initialize_parameters(self):
        init = nn.init.xavier_uniform_
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.gating1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.gating2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.gating3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.gating4 = nn.Linear(self.embedding_size, self.embedding_size)
        self.sgating1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.sgating2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.sgating3 = nn.Linear(self.embedding_size, self.embedding_size)
        self.attn = nn.Parameter(init(t.empty(1, self.embedding_size)))
        self.attn_mat = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
        
        trn_mat = self.data_handler.trn_mat
        trust_mat = self.data_handler.trust_mat
        self.trn_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(trn_mat)
        self.trust_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(trust_mat)
        self.trn_mat = self.trn_mat.coalesce().to(configs['device'])
        self.trust_mat = self.trust_mat.coalesce().to(configs['device'])
    
    def _load_data(self, trust_mat, trn_mat):
        M_matrices = self._build_motif_induced_adjacency_matrix(trust_mat, trn_mat)
        self.H_s = M_matrices[0]
        self.H_j = M_matrices[1]
        self.H_p = M_matrices[2]
        self.R = self._build_joint_adjacency(trn_mat).to(configs['device'])

    def _self_gating(self, em, channel):
        if channel == 1:
            gates = t.sigmoid(self.gating1(em))
        elif channel == 2:
            gates = t.sigmoid(self.gating2(em))
        elif channel == 3:
            gates = t.sigmoid(self.gating3(em))
        elif channel == 4:
            gates = t.sigmoid(self.gating4(em))
        return em * gates

    def _self_supervised_gating(self, em, channel):
        if channel == 1:
            sgates = t.sigmoid(self.sgating1(em))
        elif channel == 2:
            sgates = t.sigmoid(self.sgating2(em))
        elif channel == 3:
            sgates = t.sigmoid(self.sgating3(em))
        return em * sgates

    def _channel_attention(self, *channel_embeds):
        weights = []
        for embed in channel_embeds:
            weight = (self.attn * (embed @ self.attn_mat)).sum(1)
            weights.append(weight)
        weights = t.stack(weights, 0)
        score = F.softmax(t.t(weights), dim=-1)
        mixed_embeds = 0
        for i in range(len(weights)):
            mixed_embeds += t.t(t.multiply(t.t(score)[i], t.t(channel_embeds[i])))
        return mixed_embeds, score

    ######################################################
    def _compute_adamic_adar(self, trn_mat, type):
        """
        TODO: Computes the Adamic-Adar similarity matrix for either items or users based on the relation matrix.

        Args:
            relation_mat (torch sparse tensor): The relation matrix (user-item or item-user matrix)
            type (str): Type of similarity to compute ('item' or 'user')

        Returns:
            torch sparse tensor: The computed Adamic-Adar similarity matrix for users or items.
        """
        # Example implementation
        if type == 'item':
            # Compute user degree and inverse log for item similarity
            user_degree = sparse.sum(trn_mat, dim=1).to_dense()
            user_degree = 1 / t.log(user_degree + 1e-10)
            user_degree[t.isinf(user_degree)] = 0  # Set any inf values to 0
            indices = t.arange(0, user_degree.size(0)).unsqueeze(0).repeat(2, 1).to(trn_mat.device)
            user_degree_diag = t.sparse_coo_tensor(indices, user_degree, (user_degree.size(0), user_degree.size(0)))

            # Calculate item similarity matrix
            intermediate = sparse.mm(trn_mat.transpose(0, 1), user_degree_diag)
            item_similarity = sparse.mm(intermediate, trn_mat)

            # Remove diagonal values (self-similarities)
            indices = item_similarity._indices()
            values = item_similarity._values()
            mask = indices[0] != indices[1]
            new_indices = indices[:, mask]
            new_values = values[mask]
            item_similarity = t.sparse_coo_tensor(new_indices, new_values, item_similarity.shape).coalesce()

            return item_similarity

        elif type == 'user':
            # Compute item degree and inverse log for user similarity
            item_degree = sparse.sum(trn_mat, dim=0).to_dense()
            item_degree = 1 / t.log(item_degree + 1e-10)
            item_degree[t.isinf(item_degree)] = 0  # Set any inf values to 0
            indices = t.arange(0, item_degree.size(0)).unsqueeze(0).repeat(2, 1).to(trn_mat.device)
            item_degree_diag = t.sparse_coo_tensor(indices, item_degree, (item_degree.size(0), item_degree.size(0)))

            # Calculate user similarity matrix
            intermediate = sparse.mm(trn_mat, item_degree_diag)
            user_similarity = sparse.mm(intermediate, trn_mat.transpose(0, 1))

            # Remove diagonal values (self-similarities)
            indices = user_similarity._indices()
            values = user_similarity._values()
            mask = indices[0] != indices[1]
            new_indices = indices[:, mask]
            new_values = values[mask]
            user_similarity = t.sparse_coo_tensor(new_indices, new_values, user_similarity.shape).coalesce()

            return user_similarity
    ######################################################
    
    ######################################################
    def _masking_trust_matrix(self, trust_mat, uu_sim):
        """
        TODO: Masks the user-user trust matrix based on a user-user similarity threshold.

        Args:
            trust_mat (torch sparse tensor): Original user-user trust matrix (S)
            uu_sim (torch sparse tensor): User-user similarity matrix
        
        Returns:
            torch sparse tensor: Masked user-user trust matrix
        """
        device = configs['device']
        
        S = trust_mat.coalesce()
        S_values = S.values()
        min_val, max_val = S_values.min(), S_values.max()
        normalized_values = (S_values - min_val) / (max_val - min_val + 1e-10)

        sorted_indices = t.argsort(normalized_values, descending=True)

        num_samples = int(len(S_values) * configs['model']['sampling_ratio'])
        selected_indices = sorted_indices[:num_samples]

        mask_values = t.ones(num_samples).to(device) # normalization is not needed if we weight edges as 1
        masked_indices = S.indices()[:, selected_indices].to(device)
    
        return t.sparse_coo_tensor(masked_indices, mask_values, S.size()).coalesce()
    ######################################################
    
    
    ######################################################
    def _build_motif_induced_adjacency_matrix(self, trust_mat, trn_mat):
        """
        PyTorch implementation of motif-induced adjacency matrix computation.
        Args:
            trust_mat (torch sparse tensor): User-user trust matrix (S)
            trn_mat (torch sparse tensor): User-item interaction matrix (Y)
        Returns:
            List of torch sparse tensors: H_s, H_j, H_p
        """
        S = trust_mat.coalesce()
        Y = trn_mat.coalesce()
        B = S * S.transpose(0, 1)
        U = S - B

        C1 = t.sparse.mm(U, U) * U.transpose(0, 1)
        A1 = C1 + C1.transpose(0, 1)

        C2 = (t.sparse.mm(B, U) * U.transpose(0, 1)) + (t.sparse.mm(U, B) * U.transpose(0, 1)) + (t.sparse.mm(U, U) * B)
        A2 = C2 + C2.transpose(0, 1)

        C3 = (t.sparse.mm(B, B) * U) + (t.sparse.mm(B, U) * B) + (t.sparse.mm(U, B) * B)
        A3 = C3 + C3.transpose(0, 1)

        A4 = t.sparse.mm(B, B) * B

        C5 = (t.sparse.mm(U, U) * U) + (t.sparse.mm(U, U.transpose(0, 1)) * U) + (t.sparse.mm(U.transpose(0, 1), U) * U)
        A5 = C5 + C5.transpose(0, 1)

        A6 = (t.sparse.mm(U, B) * U) + (t.sparse.mm(B, U.transpose(0, 1)) * U.transpose(0, 1)) + (t.sparse.mm(U.transpose(0, 1), U) * B)

        A7 = (t.sparse.mm(U.transpose(0, 1), B) * U.transpose(0, 1)) + (t.sparse.mm(B, U) * U) + (t.sparse.mm(U, U.transpose(0, 1)) * B)

        A8 = t.sparse.mm(Y, Y.transpose(0, 1)) * B

        A9 = t.sparse.mm(Y, Y.transpose(0, 1)) * U
        A9 = A9 + A9.transpose(0, 1)

        A10 = t.sparse.mm(Y, Y.transpose(0, 1)) - A8 - A9

        H_s = A1 + A2 + A3 + A4 + A5 + A6 + A7
        H_s = H_s.coalesce()
        row_sums = t.sparse.sum(H_s, dim=1).to_dense().unsqueeze(1)
        row_sums[row_sums == 0] = 1
        H_s_values = H_s.values() / row_sums[H_s.indices()[0]].squeeze()
        H_s = t.sparse_coo_tensor(H_s.indices(), H_s_values, H_s.size())

        H_j = A8 + A9
        H_j = H_j.coalesce()
        row_sums_j = t.sparse.sum(H_j, dim=1).to_dense().unsqueeze(1)
        row_sums_j[row_sums_j == 0] = 1
        H_j_values = H_j.values() / row_sums_j[H_j.indices()[0]].squeeze()
        H_j = t.sparse_coo_tensor(H_j.indices(), H_j_values, H_j.size())

        A10_dense = A10.to_dense()
        A10_thresholded = (A10_dense > 1).float() * A10_dense
        H_p = A10_thresholded.to_sparse().coalesce()
        
        row_sums_p = t.sparse.sum(H_p, dim=1).to_dense().unsqueeze(1)
        row_sums_p[row_sums_p == 0] = 1
        H_p_values = H_p.values() / row_sums_p[H_p.indices()[0]].squeeze()
        H_p = t.sparse_coo_tensor(H_p.indices(), H_p_values, H_p.size())

        return [H_s, H_j, H_p]
    ######################################################
    
    ######################################################
    def _build_joint_adjacency(self, trn_mat):
        """
        Build a normalized joint adjacency matrix using PyTorch.
        
        Args:
            trn_mat (torch.sparse.Tensor): The user-item interaction matrix in sparse format.
            
        Returns:
            norm_adj (torch.sparse.Tensor): The normalized joint adjacency matrix.
        """
        trn_mat = trn_mat.coalesce()
        
        row_indices = trn_mat.indices()[0]
        col_indices = trn_mat.indices()[1]
        values = trn_mat.values()

        udegree = t.sparse.sum(trn_mat, dim=1).to_dense()  # User degree
        idegree = t.sparse.sum(trn_mat, dim=0).to_dense()  # Item degree

        norm_values = values / (t.sqrt(udegree[row_indices]) * t.sqrt(idegree[col_indices]))

        norm_adj = t.sparse_coo_tensor(trn_mat.indices(), norm_values, trn_mat.size())

        return norm_adj
    ######################################################

    ######################################################
    # def cal_cl_loss(self, user_embeds, ancs):
    #     """
    #     TODO: Computes the contrastive loss for user embeddings based on positive (connected) 
    #     and negative (unconnected) neighbors from the trust matrix.
        
    #     Args:
    #         user_embeds (torch tensor): User embeddings.
    #         ancs (torch tensor): Anchor users for which the contrastive loss is computed.

    #     Returns:
    #         torch scalar: Calculated contrastive loss.
    #     """
        
    #     trust_mat_new = self.trust_mat_new
    #     cl_loss = 0
        
    #     trust_row_indices = trust_mat_new.indices()[0]
    #     trust_col_indices = trust_mat_new.indices()[1]
        
    #     uu_sim_dense = self.uu_sim.to_dense()
        
    #     for u in ancs:
    #         z_u = user_embeds[u]
            
    #         # Positive neighbors (friends) based on trust_mat_new
    #         neighbors_u = trust_col_indices[trust_row_indices == u]

    #         # Inclue u itself as a component of its neighbor
    #         neighbors_u = t.cat((neighbors_u, u.unsqueeze(0)), dim=0)
            
    #         # Negative samples (non-neighbors)
    #         all_nodes = t.arange(user_embeds.size(0), device=configs['device'])
    #         non_neighbors_u = all_nodes[~t.isin(all_nodes, neighbors_u)]
    #         non_neighbors_u = non_neighbors_u[t.randperm(non_neighbors_u.size(0))[:neighbors_u.numel()]]

    #         # Compute positive similarities and numerator
    #         pos_temperature = self.temperature
    #         similarities_neighbors = t.sum(t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[neighbors_u], dim=1) / pos_temperature))
    #         numerator = similarities_neighbors

    #         # Calculate average similarity for negative samples in uu_sim matrix
    #         neg_similarities = uu_sim_dense[u, non_neighbors_u]
    #         average_neg_sim = t.mean(neg_similarities)
    #         neg_temperature = self.temperature * average_neg_sim
            
    #         # Adjust non-neighbor similarities by neg_temperature
    #         similarities_non_neighbors = t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[non_neighbors_u], dim=1) / neg_temperature)

    #         # Calculate denominator as sum of similarities with neighbors and non-neighbors
    #         denominator = t.sum(similarities_neighbors) + t.sum(similarities_non_neighbors)

    #         # Calculate -log of the probability and add to contrastive loss
    #         cl_loss_u = -t.log(numerator / (denominator + 1e-8))  # Small epsilon to prevent division by zero
    #         cl_loss += cl_loss_u

    #     return cl_loss
    def cal_cl_loss(self, user_embeds, ancs):
        """
        TODO: Computes the contrastive loss for user embeddings based on positive (connected) 
        and negative (unconnected) neighbors from the trust matrix.
        
        Args:
            user_embeds (torch tensor): User embeddings.
            ancs (torch tensor): Anchor users for which the contrastive loss is computed.

        Returns:
            torch scalar: Calculated contrastive loss.
        """
        
        trust_mat_new = self.trust_mat_new
        cl_loss = 0
        
        trust_row_indices = trust_mat_new.indices()[0]
        trust_col_indices = trust_mat_new.indices()[1]
        
        uu_sim_dense = self.uu_sim.to_dense()

        ###### 모든 u에 대한 평균 negative sample과의 similarity 정보가 있고, 이를 기준으로 정규화..? #### 좀 더 생각해보자.
        
        for u in ancs:
            z_u = user_embeds[u]
            
            # Positive neighbors (friends) based on trust_mat_new
            neighbors_u = trust_col_indices[trust_row_indices == u]

            # Inclue u itself as a component of its neighbor
            neighbors_u = t.cat((neighbors_u, u.unsqueeze(0)), dim=0)
            
            # Negative samples (non-neighbors)
            all_nodes = t.arange(user_embeds.size(0), device=configs['device'])

            ##### Changed part ##### 
            # For all samples except for positive samples as negative samples 
            non_neighbors_u = all_nodes[~t.isin(all_nodes, neighbors_u)]

            ##### Changed part ##### 
            # temperature for u: adjusted by average similarity of negative samples and u
            neg_similarities = uu_sim_dense[u, non_neighbors_u]
            average_neg_sim = t.mean(neg_similarities) # average similarity between target user and negative samples

            # 천우님 이 부분 config에 hyperparameter로 추가해주세용 k: [0.1, 0.5, 1, 5, 10]
            k = None # hyperparameter: k > 1: 더 빠르게 0.5 -> 1로 증가(평균값의 미세 변화에도 temperature 변동 up!) k<1: 천천히 증가(작은 평균값 차이에는 거의 변동 X)
            temparature_scaling =  1 / (1 + t.exp(-k * average_neg_sim))

            temperature = self.temperature * temparature_scaling

            # Compute positive similarities and numerator
            similarities_neighbors = t.sum(t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[neighbors_u], dim=1) / temperature))
            numerator = similarities_neighbors

            # Adjust non-neighbor similarities by neg_temperature
            similarities_non_neighbors = t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[non_neighbors_u], dim=1) / temperature)

            # Calculate denominator as sum of similarities with neighbors and non-neighbors
            denominator = t.sum(similarities_neighbors) + t.sum(similarities_non_neighbors)

            # Calculate -log of the probability and add to contrastive loss
            cl_loss_u = -t.log(numerator / (denominator + 1e-8))  # Small epsilon to prevent division by zero
            cl_loss += cl_loss_u

        return cl_loss


    def forward(self):
        if not self.is_training:
            return self.final_user_embeds, self.final_item_embeds
        user_embeds_c1 = self._self_gating(self.user_embeds, 1)
        user_embeds_c2 = self._self_gating(self.user_embeds, 2)
        user_embeds_c3 = self._self_gating(self.user_embeds, 3)
        simp_user_embeds = self._self_gating(self.user_embeds, 4)
        all_embeds_c1 = [user_embeds_c1]
        all_embeds_c2 = [user_embeds_c2]
        all_embeds_c3 = [user_embeds_c3]
        all_embeds_simp = [simp_user_embeds]
        item_embeds = self.item_embeds
        all_embeds_i = [item_embeds]

        for k in range(self.layer_num):
            mixed_embed = self._channel_attention(user_embeds_c1, user_embeds_c2, user_embeds_c3)[0] + simp_user_embeds / 2
            user_embeds_c1 = t.spmm(self.H_s, user_embeds_c1)
            norm_embeds = F.normalize(user_embeds_c1, p=2, dim=1)
            all_embeds_c1 += [norm_embeds]

            user_embeds_c2 = t.spmm(self.H_j, user_embeds_c2)
            norm_embeds = F.normalize(user_embeds_c2, p=2, dim=1)
            all_embeds_c2 += [norm_embeds]

            user_embeds_c3 = t.spmm(self.H_p, user_embeds_c3)
            norm_embeds = F.normalize(user_embeds_c3, p=2, dim=1)
            all_embeds_c3 += [norm_embeds]

            new_item_embeds = t.spmm(t.t(self.R), mixed_embed)
            norm_embeds = F.normalize(new_item_embeds, p=2, dim=1)
            all_embeds_i += [norm_embeds]

            simp_user_embeds = t.spmm(self.R, item_embeds)
            norm_embeds = F.normalize(simp_user_embeds, p=2, dim=1)
            all_embeds_simp += [norm_embeds]

            item_embeds = new_item_embeds

        user_embeds_c1 = sum(all_embeds_c1)
        user_embeds_c2 = sum(all_embeds_c2)
        user_embeds_c3 = sum(all_embeds_c3)
        simp_user_embeds = sum(all_embeds_simp)
        item_embeds = sum(all_embeds_i)

        ret_item_embeds = item_embeds
        ret_user_embeds, attn_score = self._channel_attention(user_embeds_c1, user_embeds_c2, user_embeds_c3)
        ret_user_embeds += simp_user_embeds / 2

        self.final_user_embeds = ret_user_embeds
        self.final_item_embeds = ret_item_embeds

        return ret_user_embeds, ret_item_embeds
    
    def _hierarchical_self_supervision(self, em, adj):
        def row_shuffle(embed):
            indices = t.randperm(embed.shape[0])
            return embed[indices]
        def row_col_shuffle(embed):
            indices = t.randperm(t.t(embed).shape[0])
            corrupted_embed = t.t(t.t(embed)[indices])
            indices = t.randperm(corrupted_embed.shape[0])
            return corrupted_embed[indices]
        def score(x1, x2):
            return (x1 * x2).sum(1)
        user_embeds = em
        edge_embeds = t.spmm(adj, user_embeds)

        pos = score(user_embeds, edge_embeds)
        neg1 = score(row_shuffle(user_embeds), edge_embeds)
        neg2 = score(row_col_shuffle(edge_embeds), user_embeds)
        local_ssl = -((pos-neg1).sigmoid().log()+(neg1-neg2).sigmoid().log()).sum()

        graph = edge_embeds.mean(0)
        pos = score(edge_embeds, graph)
        neg1 = score(row_col_shuffle(edge_embeds), graph)
        global_ssl = -(pos-neg1).sigmoid().log().sum()
        return local_ssl + global_ssl

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        
        reg_loss = self.reg_weight * reg_params(self)
        
        ss_loss = 0
        ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 1), self.H_s)
        ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 2), self.H_j)
        ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 3), self.H_p)
        ss_loss *= self.ss_rate
        
        ######################################################
        cl_loss = self.cl_weight * self.cal_cl_loss(user_embeds, ancs)
        ######################################################
        
        
        loss = bpr_loss + reg_loss + ss_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'ss_loss': ss_loss, 
                  'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
