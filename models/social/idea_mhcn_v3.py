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
        
        self.uu_sim = self._compute_adamic_adar(self.trn_mat)
        self.trust_mat_new = self._masking_trust_matrix(self.trust_mat, self.uu_sim)
        self.social_dict = self._get_social_dict(self.trust_mat, self.uu_sim) # dictonary of (user: neighbors, temperature_scaling_factor)
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
    def _compute_adamic_adar(self, trn_mat):
        """
        TODO: Computes the Adamic-Adar similarity matrix for either items or users based on the relation matrix.

        Args:
            trn_mat (torch sparse tensor): The relation matrix (user-item or item-user matrix)

        Returns:
            torch sparse tensor: The computed Adamic-Adar similarity matrix for users or items.
        """
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
        
        # Coalesce matrices to ensure indices and values align
        S = trust_mat.coalesce()
        S_indices = S.indices()
        uu_sim_coo = uu_sim.coalesce()
        
        # Step 1: Convert sparse matrices to dense
        S_dense = trust_mat.to_dense().to(device)
        uu_sim_dense = uu_sim.to_dense().to(device)
        
        # Step 2: Perform element-wise multiplication
        multiplied_dense = S_dense * uu_sim_dense

        # Step 3: Convert to sparse format
        multiplied_sparse = multiplied_dense.to_sparse().coalesce()
        
        # Step 4: Sort the non-zero values in the sparse matrix
        multiplied_values = multiplied_sparse.values()
        sorted_indices = t.argsort(multiplied_values, descending=True)
        
        # Calculate the number of samples based on sampling_ratio
        num_samples = int(len(multiplied_values) * configs['model']['sampling_ratio'])
        selected_indices = sorted_indices[:num_samples]

        # Step 5: Select indices and values for the sampled entries
        mask_values = multiplied_values[selected_indices]
        mask_indices = multiplied_sparse.indices()[:, selected_indices].to(device)

        # Return the final sparse tensor with selected top values
        return t.sparse_coo_tensor(mask_indices, mask_values, multiplied_dense.size()).coalesce()
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
    def _get_social_dict(self, trust_mat, uu_sim):
        """
        Computes a dictionary for each user containing their positive neighbors and 
        a temperature scaling factor based on the average similarity with non-neighbors (negative samples).

        Args:
            trust_mat (torch sparse tensor): The user-user trust matrix.
            uu_sim (torch sparse tensor): The user-user similarity matrix.

        Returns:
            dict: A dictionary where keys are user indices and values are dictionaries containing:
                - 'neighbors': The user's neighbors (including the user itself).
                - 'temperature_scaling_factor': The calculated temperature scaling factor.
        """
        
        # Extract row and column indices of non-zero elements in the trust matrix
        trust_row_indices = trust_mat.indices()[0]
        trust_col_indices = trust_mat.indices()[1]
        
        # Convert similarity matrix to dense for easier indexing
        uu_sim_dense = uu_sim.to_dense()
        
        # Initialize an empty dictionary to store neighbors and scaling factors for each user
        social_dict = {}
        users = t.arange(uu_sim.size(0), device=configs['device'])
        k = configs['model']['k']  # Hyperparameter for scaling adjustment sensitivity
        
        # Iterate over each user to compute their neighbors and scaling factor
        for u in users:
            # Find positive neighbors of user `u` in the trust matrix and add `u` itself
            neighbors_u = trust_col_indices[trust_row_indices == u]
            neighbors_u = t.cat((neighbors_u, u.unsqueeze(0)), dim=0)
            
            # Identify non-neighbors (negative samples) by excluding neighbors from all users
            non_neighbors_u = users[~t.isin(users, neighbors_u)]
            
            # Compute the average similarity with negative samples
            neg_similarities = uu_sim_dense[u, non_neighbors_u]
            average_neg_sim = t.mean(neg_similarities)
            
            # Calculate the temperature scaling factor with a sigmoid function
            temperature_scaling_factor = 1 / (1 + t.exp(-k * average_neg_sim))
            
            # Store the neighbors and scaling factor in a dictionary for this user
            social_dict[u.item()] = {
                'neighbors': neighbors_u, 
                'temperature_scaling_factor': temperature_scaling_factor
            }
        
        return social_dict
    ######################################################
    
    
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
        
        # trust_mat_new = self.trust_mat_new
        cl_loss = 0
        
        # Move indices to CPU
        # trust_row_indices = trust_mat_new.indices()[0].to('cpu')
        # trust_col_indices = trust_mat_new.indices()[1].to('cpu')
        
        # Move uu_sim and user_embeds to CPU
        # uu_sim_dense = self.uu_sim.to_dense().to('cpu')
        user_embeds_cpu = user_embeds.to('cpu')  # Move user embeddings to CPU for this computation
        all_nodes = t.arange(user_embeds_cpu.size(0), device='cpu')
        
        # Ensure ancs is on CPU
        ancs_cpu = ancs.to('cpu')

        social_dict = self.social_dict
        
        for u in ancs:
            u = u.item()
            z_u = user_embeds[u]
            
            temperature_scaling = social_dict[u]['temperature_scaling_factor']
            temperature = self.temperature * temperature_scaling
            

            neighbors_u = social_dict[u]['neighbors'].to('cpu')
            
            # non_neighbors_u = all_nodes[~t.isin(all_nodes_cpu, neighbors_u)] # for all nodes - neighbors
            non_neighbors_u = ancs_cpu[~t.isin(ancs_cpu, neighbors_u)] # for ancs - neighbors

            # Compute positive similarities and numerator
            similarities_neighbors = t.sum(t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[neighbors_u], dim=1) / temperature))
            numerator = similarities_neighbors

            # Adjust non-neighbor similarities by neg_temperature
            similarities_non_neighbors = t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[non_neighbors_u], dim=1) / temperature)

            # Calculate denominator as sum of similarities with neighbors and non-neighbors
            denominator = t.sum(similarities_neighbors) + t.sum(similarities_non_neighbors)

            # Calculate -log of the probability and add to contrastive loss
            cl_loss_u = -temperature * t.log(numerator / (denominator + 1e-8))  # Small epsilon to prevent division by zero
            cl_loss += cl_loss_u

        print(1)
        exit()

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
