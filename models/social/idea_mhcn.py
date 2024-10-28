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
######################################################

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class IDEA_MHCN(BaseModel):
    def __init__(self, data_handler):
        super(IDEA_MHCN, self).__init__(data_handler)
        self.data_handler = data_handler
        self._load_configs()
        self._initialize_parameters()
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
    def _compute_one_homophily(item_embeds, items_A, items_B):
        """
        Compute homophily between two users (userA and userB) based on their connected items.
        
        Args:
            item_embeds (torch.Tensor): Item embeddings matrix (shape: [num_items, embedding_dim]). This matrix is used to compute similarity between items based on user interactions.
            items_A (list of int): List of item indices connected to userA.
            items_B (list of int): List of item indices connected to userB.
        
        Returns:
            float: Homophily value between items_A and items_B
        """
        
        # Find unique items in each set (non-overlapping items)
        intersection_size = len(set(items_A).intersection(set(items_B)))
        union_size = len(set(items_A).union(set(items_B)))
        
        unique_A = list(set(items_A) - set(items_B))
        unique_B = list(set(items_B) - set(items_A))
        
        # TODO: For each item in unique_A, find the maximum inner product with all items in unique_B and sum these values. Similarly, for each item in unique_B, find the maximum inner product with all items in unique_A and sum these values
        max_similarities_A = ###
        max_similarities_B = ###

        numerator = intersection_size + max_similarities_A + max_similarities_B
        denominator = union_size 
        
        homophily_value = numerator / denominator
        return homophily_value
    ######################################################



    ######################################################
    def _compute_homophily_ratios(self, item_embeds):
        """
        Compute edge-wise homophily ratios using item embeddings for user pairs in the trust matrix(social graph).
    
        Args:
        item_embeds (torch.Tensor): Item embeddings matrix (shape: [num_items, embedding_dim]). These embeddings are used to compute similarity between users based on their interactions with items.

        Returns:
        homophily_ratios (torch.Tensor): Tensor containing homophily ratios (similarities) for each edge in the social graph based on user-item interactions.
        """

        trust_mat = self.trust_mat.coalesce()
        row_indices = trust_mat.indices()[0]
        col_indices = trust_mat.indices()[1]
        homophily_ratios = []
        
        for userA, userB in tqdm(zip(row_indices, col_indices), desc='Computing Homophily', total=len(row_indices)):
            items_A = self.data_handler.get_connected_items(userA.item())
            items_B = self.data_handler.get_connected_items(userB.item())
            
            similarity = self._compute_one_homophily(item_embeds, items_A, items_B) # TODO: complete this function
            
            homophily_ratios.append(similarity)
        
        return t.tensor(homophily_ratios)
    ######################################################

    ######################################################
    def _compute_cosine_similarity(self, user_embeds):
        """
        Compute cosine similarity between user embeddings.
        Args:
            user_embeds (torch.Tensor): User embeddings (z_u^(l+1))
        Returns:
            similarity_matrix (torch.Tensor): Cosine similarity matrix between users
        """
        user_embeds_norm = F.normalize(user_embeds, p=2, dim=1)
        similarity_matrix = t.matmul(user_embeds_norm, user_embeds_norm.T)
        
        return similarity_matrix
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
    def _rewire_social_graph(self):
        """
        Rewires the social graph based on the similarity between user embeddings.
        This is a placeholder function, and the rewiring strategy can be adjusted.
        
        Returns:
            torch.sparse.Tensor: A new trust matrix with updated edge weights based on homophily ratios.
        """
        # Retrieve the existing trust matrix and user/item embeddings
        trust_mat = self.trust_mat
        user_embeds = self.user_embeds
        item_embeds = self.item_embeds
        
        # Compute homophily ratios based on item embeddings for each user pair in the trust matrix
        self.homophily_ratios = self._compute_homophily_ratios(item_embeds)
        
        # Get indices of existing edges in the trust matrix
        indices = trust_mat.indices()
        row_indices = indices[0].long()  # Source user indices
        col_indices = indices[1].long()  # Target user indices
        
        new_values = []
        for i, j in zip(row_indices, col_indices):
            # TODO: make social graph with weighted edges based on homophily ratios
            
            new_value = ###
            new_values.append(new_value)

        new_trust_mat = t.sparse_coo_tensor(indices, new_values, trust_mat.size(), dtype=t.float32)
        
        return new_trust_mat
    ######################################################


    ######################################################
    def cal_cl_loss(self, user_embeds, ancs):
        """
        Calculate the MIL NCE loss based on homophily ratios and InfoNCE.
        
        Args:
            user_embeds (torch.Tensor): User embeddings.
            ancs (torch.Tensor): Ancestors or reference nodes for calculating contrastive loss.
            
        Returns:
            cl_loss (torch.Tensor): Contrastive learning loss 
        """
        # TODO: check the semantic of CL loss
        
        device = configs['device']
        homophily_ratios = self.homophily_ratios.to(device)
        
        trust_mat = self.trust_mat.coalesce()
        row_indices = trust_mat.indices()[0].to(device) 
        col_indices = trust_mat.indices()[1].to(device)
        
        cl_loss = 0

        for u in ancs:
            z_u = user_embeds[u] 
            neighbors_u = col_indices[row_indices == u]
            
            # Skip if no neighbors are present
            if neighbors_u.numel() == 0:
                continue  
            
            # Select the same number of non-neighbors as neighbors_u
            all_nodes = t.arange(user_embeds.size(0), device=device)
            non_neighbors_u = all_nodes[~t.isin(all_nodes, neighbors_u)]
            non_neighbors_u = non_neighbors_u[t.randperm(non_neighbors_u.size(0))[:neighbors_u.numel()]]

            # Calculate homophily-weighted similarity for each neighbor in neighbors_u
            homophily_u = homophily_ratios[row_indices == u]
            similarities_neighbors = t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[neighbors_u], dim=1))
            numerator = t.sum(homophily_u * similarities_neighbors)

            # Calculate similarity for each non-neighbor in non_neighbors_u
            similarities_non_neighbors = t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[non_neighbors_u], dim=1))

            # Calculate denominator as sum of similarities with neighbors and non-neighbors
            denominator = t.sum(similarities_neighbors) + t.sum(similarities_non_neighbors)

            # Calculate -log of the probability and add to contrastive loss
            cl_loss_u = -t.log(numerator / (denominator + 1e-8))  # Small epsilon to prevent division by zero
            cl_loss += cl_loss_u

        return cl_loss
    ######################################################




    def forward(self):
        if not self.is_training:
            return self.final_user_embeds, self.final_item_embeds
        ######################################################
        else:
            trust_mat = self._rewire_social_graph()
            trn_mat = self.trn_mat

            M_matrices = self._build_motif_induced_adjacency_matrix(trust_mat, trn_mat)

            self.data_handler.H_s = M_matrices[0]
            self.data_handler.H_j = M_matrices[1]
            self.data_handler.H_p = M_matrices[2]
            
            self.data_handler.R = self._build_joint_adjacency(trn_mat)
        ######################################################
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

            user_embeds_c1 = t.spmm(self.data_handler.H_s, user_embeds_c1)
            norm_embeds = F.normalize(user_embeds_c1, p=2, dim=1)
            all_embeds_c1 += [norm_embeds]

            user_embeds_c2 = t.spmm(self.data_handler.H_j, user_embeds_c2)
            norm_embeds = F.normalize(user_embeds_c2, p=2, dim=1)
            all_embeds_c2 += [norm_embeds]

            user_embeds_c3 = t.spmm(self.data_handler.H_p, user_embeds_c3)
            norm_embeds = F.normalize(user_embeds_c3, p=2, dim=1)
            all_embeds_c3 += [norm_embeds]

            new_item_embeds = t.spmm(t.t(self.data_handler.R), mixed_embed)
            norm_embeds = F.normalize(new_item_embeds, p=2, dim=1)
            all_embeds_i += [norm_embeds]

            simp_user_embeds = t.spmm(self.data_handler.R, item_embeds)
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
        ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 1), self.data_handler.H_s)
        ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 2), self.data_handler.H_j)
        ss_loss += self._hierarchical_self_supervision(self._self_supervised_gating(user_embeds, 3), self.data_handler.H_p)
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