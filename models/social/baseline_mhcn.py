import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, cal_infonce_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
######################################################
import numpy as np
from torch.utils.checkpoint import checkpoint
######################################################

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class BASELINE_MHCN(BaseModel):
    def __init__(self, data_handler):
        super(BASELINE_MHCN, self).__init__(data_handler)
        self.data_handler = data_handler
        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        self.ss_rate = configs['model']['ss_rate']
        ######################################################
        self.device = configs['device']
        self.cl_weight = configs['model']['cl_weight']
        self.zeta = configs['model']['zeta']
        self.temperature = configs['model']['temperature']
        self.homophily_ratios = self._compute_homophily_ratios()
        ######################################################
        
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
        self.is_training = True

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
    def _compute_homophily_ratios(self):
        """
        Compute edge-wise homophily ratios using Jaccard similarity between user neighbors.
        Returns:
            homophily_ratios (torch.Tensor): Homophily ratios for each edge in the social graph
        """

        trust_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(self.data_handler.trust_mat)
        trust_mat = trust_mat.coalesce() 
        row_indices = trust_mat.indices()[0]
        col_indices = trust_mat.indices()[1]
        
        user_neigh = {}

        for u in t.unique(row_indices):  # Iterate over unique users
            user_neigh[u.item()] = set(trust_mat.indices()[1][trust_mat.indices()[0] == u].tolist())

        # Vectorized Jaccard similarity calculation between neighbors of user pairs
        homophily_ratios = []
        
        for i, j in zip(row_indices, col_indices):
            neighbors_i = user_neigh[i.item()]
            neighbors_j = user_neigh[j.item()]
            
            # Compute intersection and union sizes for Jaccard similarity
            intersection_size = len(neighbors_i & neighbors_j)
            union_size = len(neighbors_i | neighbors_j)
            jaccard_similarity = intersection_size / union_size if union_size != 0 else 0.0
            homophily_ratios.append(jaccard_similarity)
        
        return t.tensor(homophily_ratios).to(self.device)
    ######################################################

    ######################################################
    def _encode_user_embeddings(self):
        """
        Compute user embeddings based on the user-item interaction matrix using the 
        formula for SGR. This corresponds to the encoder in the SHaRe framework.
        """
        interaction_matrix = self.data_handler.trn_mat
        interaction_matrix = self.data_handler._sparse_mx_to_torch_sparse_tensor(interaction_matrix).coalesce()
        row_indices = interaction_matrix.indices()[0]
        col_indices = interaction_matrix.indices()[1]
        values = interaction_matrix.values()
        
        all_item_embeds = [self.item_embeds]
        all_user_embeds = [self.user_embeds]
        
        user_degrees = interaction_matrix.sum(axis=1).to_dense()
        item_degrees = interaction_matrix.sum(axis=0).to_dense()
        
        inv_sqrt_user_degrees = 1.0 / t.sqrt(user_degrees + 1e-8)
        inv_sqrt_item_degrees = 1.0 / t.sqrt(item_degrees + 1e-8)
        
        normalized_values = values * inv_sqrt_user_degrees[row_indices] * inv_sqrt_item_degrees[col_indices]
        interaction_matrix_norm_tensor = t.sparse_coo_tensor(interaction_matrix.indices(), normalized_values, interaction_matrix.size())

        
        for layer in range(self.layer_num):
            
            user_embeds = t.sparse.mm(interaction_matrix_norm_tensor, all_item_embeds[layer])
            all_user_embeds.append(user_embeds)
            item_embeds = t.sparse.mm(interaction_matrix_norm_tensor.transpose(0, 1), all_user_embeds[layer])
            all_item_embeds.append(item_embeds)

        user_embeds_concat = t.cat(all_user_embeds, dim=1)
        
        return user_embeds_concat

    ######################################################
    def _compute_cosine_similarity(self, user_embeds, user_pairs):
        """
        Compute cosine similarity between specific pairs of user embeddings.
        Args:
            user_embeds (torch.Tensor): User embeddings (shape: [user_num, embed_dim])
            user_pairs (torch.Tensor): Indices of user pairs (shape: [2, num_pairs])
        Returns:
            similarities (torch.Tensor): Cosine similarities for the specified pairs
        """
        with t.no_grad():
            u_indices = user_pairs[0]
            v_indices = user_pairs[1]
            
            u_embeds = user_embeds[u_indices]
            v_embeds = user_embeds[v_indices]
            
            u_embeds_norm = F.normalize(u_embeds, p=2, dim=1)
            v_embeds_norm = F.normalize(v_embeds, p=2, dim=1)
            
            similarities = (u_embeds_norm * v_embeds_norm).sum(dim=1)
        return similarities
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

        # Compute H_s
        C1 = t.sparse.mm(U, U) * U.transpose(0, 1)
        A1 = C1 + C1.transpose(0, 1)
        # del C1  # Free up memory

        C2 = (t.sparse.mm(B, U) * U.transpose(0, 1)) + (t.sparse.mm(U, B) * U.transpose(0, 1)) + (t.sparse.mm(U, U) * B)
        A2 = C2 + C2.transpose(0, 1)
        # del C2  # Free up memory

        C3 = (t.sparse.mm(B, B) * U) + (t.sparse.mm(B, U) * B) + (t.sparse.mm(U, B) * B)
        A3 = C3 + C3.transpose(0, 1)
        # del C3  # Free up memory

        A4 = t.sparse.mm(B, B) * B

        C5 = (t.sparse.mm(U, U) * U) + (t.sparse.mm(U, U.transpose(0, 1)) * U) + (t.sparse.mm(U.transpose(0, 1), U) * U)
        A5 = C5 + C5.transpose(0, 1)
        # del C5  # Free up memory

        A6 = (t.sparse.mm(U, B) * U) + (t.sparse.mm(B, U.transpose(0, 1)) * U.transpose(0, 1)) + (t.sparse.mm(U.transpose(0, 1), U) * B)

        A7 = (t.sparse.mm(U.transpose(0, 1), B) * U.transpose(0, 1)) + (t.sparse.mm(B, U) * U) + (t.sparse.mm(U, U.transpose(0, 1)) * B)

        H_s = A1 + A2 + A3 + A4 + A5 + A6 + A7
        H_s = H_s.coalesce()
        row_sums = t.sparse.sum(H_s, dim=1).to_dense().unsqueeze(1)
        row_sums[row_sums == 0] = 1
        H_s_values = H_s.values() / row_sums[H_s.indices()[0]].squeeze()
        H_s = t.sparse_coo_tensor(H_s.indices(), H_s_values, H_s.size())
        
        # Free up memory used by A1 ~ A7
        # del A1, A2, A3, A4, A5, A6, A7
        # del row_sums, H_s_values

        # Compute H_j
        A8 = t.sparse.mm(Y, Y.transpose(0, 1)) * B
        A9 = t.sparse.mm(Y, Y.transpose(0, 1)) * U
        A9 = A9 + A9.transpose(0, 1)
        A10 = t.sparse.mm(Y, Y.transpose(0, 1)) - A8 - A9
        
        H_j = A8 + A9
        H_j = H_j.coalesce()
        row_sums_j = t.sparse.sum(H_j, dim=1).to_dense().unsqueeze(1)
        row_sums_j[row_sums_j == 0] = 1
        H_j_values = H_j.values() / row_sums_j[H_j.indices()[0]].squeeze()
        H_j = t.sparse_coo_tensor(H_j.indices(), H_j_values, H_j.size())

        # Free up memory used by A8, A9
        # del A8, A9, row_sums_j, H_j_values

        # Compute H_p
        # A10_dense = A10.to_dense()
        # A10_thresholded = (A10_dense > 1) * A10_dense
        # H_p = A10_thresholded.to_sparse().coalesce()
        
        A10 = A10.coalesce()
        indices = A10.indices()
        values = A10.values()
        mask = values > 1
        indices = indices[:, mask]
        values = values[mask]
        H_p = t.sparse_coo_tensor(indices, values, A10.size()).coalesce()
        
        row_sums_p = t.sparse.sum(H_p, dim=1).to_dense().unsqueeze(1)
        row_sums_p[row_sums_p == 0] = 1
        H_p_values = H_p.values() / row_sums_p[H_p.indices()[0]].squeeze()
        H_p = t.sparse_coo_tensor(H_p.indices(), H_p_values, H_p.size())

        # Return the final matrices
        return [H_s, H_j, H_p]

    ######################################################
    
    ######################################################
    def _sample_potential_edges(self, num_samples, existing_pairs, user_num):
        if num_samples == 0:
            return t.empty((2, 0), dtype=t.long).to(self.device)
        
        existing_set = set(zip(existing_pairs[0].tolist(), existing_pairs[1].tolist()))
        potential_pairs = []
        while len(potential_pairs) < num_samples:
            u = t.randint(0, user_num, (num_samples,))
            v = t.randint(0, user_num, (num_samples,))
            mask = u != v
            u = u[mask]
            v = v[mask]
            pairs = list(zip(u.tolist(), v.tolist()))
            new_pairs = [pair for pair in pairs if pair not in existing_set]
            potential_pairs.extend(new_pairs)
            potential_pairs = potential_pairs[:num_samples]
        potential_pairs = t.tensor(potential_pairs).transpose(0, 1)
        return potential_pairs.to(self.device)
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
        """
        trust_mat = self.data_handler.trust_mat
        new_trust_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(trust_mat).coalesce()
        
        
        user_embeds = self._encode_user_embeddings()
        # user_embeds = self.user_embeds.detach()
        
        indices = new_trust_mat.indices()
        row_indices = indices[0].long()
        col_indices = indices[1].long()
        
        # Compute similarities only for existing edges
        existing_pairs = t.stack([row_indices, col_indices], dim=0)
        similarity_values_at_trust_edges = self._compute_cosine_similarity(user_embeds, existing_pairs)
    
        cut_edge_flags = similarity_values_at_trust_edges <= 0
        num_cut_edges = int(cut_edge_flags.sum().item())  
        
        
        
        new_trust_mat_values = new_trust_mat.values().clone()
        new_trust_mat_values[cut_edge_flags] = 0
        
        if num_cut_edges == 0:
            # No edges to cut, return the updated trust_mat
            new_trust_mat = t.sparse_coo_tensor(indices, new_trust_mat_values, new_trust_mat.size(), dtype=t.float32)
            return new_trust_mat.coalesce()
        
        user_num = user_embeds.shape[0]
        potential_pairs = self._sample_potential_edges(num_cut_edges, existing_pairs, user_num)
        
        # Compute similarities for potential new edges
        potential_similarities = self._compute_cosine_similarity(user_embeds, potential_pairs)
        top_values, top_indices = t.topk(potential_similarities, num_cut_edges)
        top_pairs = potential_pairs[:, top_indices]

        all_indices = t.cat([indices[:, ~cut_edge_flags], top_pairs], dim=1)
        all_values = t.cat([new_trust_mat_values[~cut_edge_flags], top_values], dim=0)
        
        new_trust_mat = t.sparse_coo_tensor(all_indices, all_values, new_trust_mat.size(), dtype=t.float32)
        
        return new_trust_mat.coalesce()
    ######################################################

    ######################################################
    def _get_interaction_graph(self):
        trn_mat = self.data_handler.trn_mat
        if not isinstance(trn_mat, t.Tensor):
            trn_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(trn_mat)
        return trn_mat
    ######################################################

    ######################################################
    def cal_mil_nce_loss(self, user_embeds, ancs):
        """
        Calculate the MIL NCE loss based on homophily ratios and InfoNCE.
        Args:
            user_embeds (torch.Tensor): User embeddings
        Returns:
            cl_loss (torch.Tensor): Contrastive learning loss (MIL NCE)
        """
        device = configs['device']
        homophily_ratios = self.homophily_ratios.to(device)
        
        h_min, h_max = homophily_ratios.min(), homophily_ratios.max()
        
        epsilon = (self.zeta - h_min) / (h_max - h_min + 1e-8) 
        
        
        trust_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(self.data_handler.trust_mat).coalesce()
        row_indices = trust_mat.indices()[0].to(device) 
        col_indices = trust_mat.indices()[1].to(device)
        
        cl_loss = 0

        for u in ancs:
            z_u = user_embeds[u] 
            neighbors_u = col_indices[row_indices == u] 
            
            if neighbors_u.numel() == 0:
                continue  
            
            homophily_u = homophily_ratios[row_indices == u] 
            pos_samples = neighbors_u[homophily_u > epsilon] 
            neg_samples = neighbors_u[homophily_u <= epsilon] 
            
            if pos_samples.numel() == 0 or neg_samples.numel() == 0:
                continue
            
            pos_sum = t.sum(t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[pos_samples], dim=1) / self.temperature))
            neg_sum = t.sum(t.exp(t.cosine_similarity(z_u.unsqueeze(0), user_embeds[neg_samples], dim=1) / self.temperature))
            
            cl_loss_u = -t.log(pos_sum / (pos_sum + neg_sum + 1e-8))  
            cl_loss += cl_loss_u  
            

        

        return cl_loss
    ######################################################




    def forward(self):
        if not self.is_training:
            return self.final_user_embeds, self.final_item_embeds
        ######################################################
        else:
            # trust_mat = checkpoint(self._rewire_social_graph, use_reentrant=False)
            trust_mat = self._rewire_social_graph()
            trn_mat = self._get_interaction_graph()

            # M_matrices = checkpoint(self._build_motif_induced_adjacency_matrix, trust_mat, trn_mat, use_reentrant=False)
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
        cl_loss = self.cl_weight * self.cal_mil_nce_loss(user_embeds, ancs)
        ######################################################
        
        
        loss = bpr_loss + reg_loss + ss_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'ss_loss': ss_loss, 
                  'cl_loss': cl_loss}
        
        # loss = bpr_loss + reg_loss + ss_loss
        # losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'ss_loss': ss_loss}
        
        return loss, losses

    def full_predict(self, batch_data):
        with t.no_grad():
            user_embeds, item_embeds = self.forward()
            self.is_training = False
            pck_users, train_mask = batch_data
            pck_users = pck_users.long()
            pck_user_embeds = user_embeds[pck_users]
            full_preds = pck_user_embeds @ item_embeds.T
            full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds