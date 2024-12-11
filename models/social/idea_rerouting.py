import torch as t
from torch import nn
from torch import sparse
from models.base_model import BaseModel
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
import networkx as nx
from models.model_utils import SelfGatingUnit

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class IDEA_REROUTING(BaseModel):
    def __init__(self, data_handler):
        super(IDEA_REROUTING, self).__init__(data_handler)

        self.device = configs['device']
        
        # Data handler에서 trn_mat과 trust_mat 가져오기
        self.trn_mat = self._coo_to_sparse_tensor(data_handler.trn_mat)
        self.trust_mat = self._coo_to_sparse_tensor(data_handler.trust_mat)

        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))


        # applying rerouting
        self.uu_sim = self._compute_adamic_adar(self.trn_mat)

        threshold = configs['model']['similarity_threshold']
        # self.trust_mat_new = self._reroute_trust_matrix(self.trust_mat, self.uu_sim, threshold)
        self.trust_mat_new = self.trust_mat.clone()

        # what should be set here?
        # self.trust_mat = self.trust_mat_new


        self.is_training = True
        self.final_embeds = None
        
        # COO Matrix -> Sparse Tensor 변환 및 정규화
        A = self._create_adj_matrix(self.trn_mat)
        self.adj = self._normalize_sparse_matrix(A)
        
        if 'self_gating_unit' in configs['model'] and configs['model']['self_gating_unit']:
            self.self_gating_unit_social = SelfGatingUnit(self.embedding_size)
            self.self_gating_unit_interaction = SelfGatingUnit(self.embedding_size)
            # usage example:
            # embedding_after_sgl = self.self_gating_unit(embedding_before_sgl)
            # user_embeds_social = self.self_gating_unit(user_embeds)
        
        if 'pagerank' in configs['model'] and configs['model']['pagerank']:
            pagerank = self._get_pagerank(self.trust_mat, self.user_num) # t.tensor
            self.pagerank_normalized_trust_matrix = self._pagerank_normalized_trust_matrix(pagerank)


    def _reroute_trust_matrix(self, trust_mat, uu_sim, threshold_quantile):
        device = configs["device"]

        S_dense = trust_mat.to_dense().to(device)
        uu_sim_dense = uu_sim.to_dense().to(device)

        connected_sim = uu_sim_dense[S_dense == 1]
        threshold = connected_sim.quantile(threshold_quantile).item()

        # weak connection is connected in S_dense but low uu_sim_dense
        weak_connections = (S_dense == 1) & (uu_sim_dense < threshold)

        # print(f"Number of weak connections: {weak_connections.sum().item()}")

        weak_connections = weak_connections.nonzero()

        for i, j in weak_connections:

            # get indices of neighbors of i
            idx_Ni = (S_dense[i] == 1).nonzero(as_tuple=False)
            
            # checking if the dimension is above 1
            if idx_Ni.dim() > 1:
                idx_Ni = idx_Ni.squeeze()

            # similarities between neighbors of i and j
            if idx_Ni.numel() > 0:

                if len(idx_Ni.shape) == 0:
                    idx_Ni = idx_Ni.unsqueeze(0)

                Ni_sim_j = uu_sim_dense[idx_Ni, j]
                max_sim_Ni_idx = Ni_sim_j.argmax()

                max_sim_Ni = Ni_sim_j.max()
                max_Ni = idx_Ni[max_sim_Ni_idx] # the neighbor of i with the highest similarity to j
            else:
                max_sim_Ni = -float('inf')
                max_Ni = None

            # get indices of neighbors of j
            idx_Nj = (S_dense[j] == 1).nonzero(as_tuple=False)

            # checking if the dimension is above 1
            if idx_Nj.dim() > 1:
                idx_Nj = idx_Nj.squeeze()

            # similarities between neighbors of j and i
            if idx_Nj.numel() > 0:

                if len(idx_Nj.shape) == 0:
                    idx_Nj = idx_Nj.unsqueeze(0)

                Nj_sim_i = uu_sim_dense[i, idx_Nj]
                max_sim_Nj_idx = Nj_sim_i.argmax()

                max_sim_Nj = Nj_sim_i.max()
                max_Nj = idx_Nj[max_sim_Nj_idx]
            else:
                max_sim_Nj = -float('inf')
                max_Nj = None

            # check if the neighbors are better than the original
            sim_i_j = uu_sim_dense[i, j]
            if sim_i_j < max_sim_Ni or sim_i_j < max_sim_Nj:
                # reroute

                # remove the weak edge between i and j
                S_dense[i, j] = 0
                S_dense[j, i] = 0

                # decide on rerouting
                if max_sim_Ni > max_sim_Nj and max_Ni is not None:
                    # reroute through neighbor of i

                    S_dense[max_Ni, j] = uu_sim_dense[max_Ni, j]  # or 1
                    S_dense[j, max_Ni] = uu_sim_dense[max_Ni, j]  

                elif max_sim_Nj >= max_sim_Ni and max_Nj is not None:
                    # reroute through neighbor of j

                    S_dense[i, max_Nj] = 1  # Or use nj_i_sim as weight
                    S_dense[max_Nj, i] = 1  # if undirected

                else:
                    pass

        S = S_dense.to_sparse().coalesce()
        return S



    def _compute_adamic_adar(self, trn_mat):
        """
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



    def _get_pagerank(self, trust_mat, num_users, alpha=0.85, max_iter=100, tol=1e-6):
        """
        Compute the PageRank of nodes in a social graph using networkx.
        
        Args:
            trust_mat (t.sparse_coo_tensor): Sparse adjacency matrix of the social graph.
            num_users (int): Number of users in the social graph.
            alpha (float): Damping factor (usually 0.85).
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
        
        Returns:
            t.Tensor: PageRank scores for each user.
        """
        
        edge_index = trust_mat._indices()
        edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        G = nx.DiGraph()
        G.add_edges_from(edge_list)
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
        
        pagerank_full = {i: pagerank.get(i, 0.0) for i in range(num_users)} # some users do not have relation in social graph
        pagerank = t.tensor([pagerank_full[i] for i in range(num_users)], dtype=t.float32, device=self.device)
        
        # sorted_indices = t.argsort(pr, descending=True)
        # top_3 = sorted_indices[:3]
        # print(f"Top 3 PageRank scores for {configs['data']['name']} dataset:")
        # for idx in top_3:
        #     print(f"Node {idx.item()}: {pr_scores[idx].item():.5f}")
        # print(f"Average PageRank score: {1/num_users:.5f}")
        # exit()
        
        return pagerank
        
    ######################################################
    def _pagerank_normalized_trust_matrix(self, pagerank):
        pagerank_sqrt = t.sqrt(pagerank)
        normalized_matrix = t.ger(pagerank_sqrt, pagerank_sqrt) # outer product
        return normalized_matrix
    ######################################################
    
    
    def _coo_to_sparse_tensor(self, coo_mat):
        """COO matrix를 torch sparse tensor로 변환"""
        coo_mat = coo_mat.tocoo()
        indices = t.stack([
            t.from_numpy(coo_mat.row).long(),
            t.from_numpy(coo_mat.col).long()
        ])
        values = t.from_numpy(coo_mat.data).float()
        shape = coo_mat.shape
        return t.sparse_coo_tensor(indices, values, shape, device=self.device).coalesce()

    def _create_adj_matrix(self, trn_mat):
        """PyTorch sparse tensor로부터 A 행렬 생성 및 정규화"""
        user_num = self.user_num
        item_num = self.item_num
        total_num = user_num + item_num

        # trn_mat: (user_num, item_num) 크기의 sparse tensor
        trn_indices = trn_mat._indices()
        trn_values = trn_mat._values()

        # user-item interaction 추가
        user_indices = trn_indices[0, :]
        item_indices = trn_indices[1, :] + user_num  # item index shift

        # (user, item) -> (user, user_num + item)
        upper_indices = t.stack([user_indices, item_indices])
        lower_indices = t.stack([item_indices, user_indices])

        # 상단과 하단 모두에 값 추가
        combined_indices = t.cat([upper_indices, lower_indices], dim=1)
        combined_values = t.cat([trn_values, trn_values], dim=0)

        # (user + item, user + item) 크기의 A matrix 생성
        A = t.sparse_coo_tensor(combined_indices, combined_values, (total_num, total_num), device=self.device).coalesce()
        return A

    def _normalize_sparse_matrix(self, mat):
        """Sparse Matrix (torch)를 대칭 정규화하는 함수"""
        # 행 합 계산 (degree)
        degree = t.sparse.sum(mat, dim=1).to_dense()
        degree_inv_sqrt = t.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0  # 무한대 값 처리

        # 정규화된 행렬 계산: \mathbf{D}^{-1/2} * \mathbf{A} * \mathbf{D}^{-1/2}
        values = mat.values()
        indices = mat.indices()

        row = indices[0]
        col = indices[1]

        # 정규화 값 계산
        norm_values = degree_inv_sqrt[row] * values * degree_inv_sqrt[col]
        normalized_mat = t.sparse_coo_tensor(indices, norm_values, mat.size(), device=self.device)

        return normalized_mat.coalesce()

    def _create_combined_adj(self, adj, trust_adj):
        """
        User-Item Graph와 Social Graph를 결합한 전파 행렬 생성.
        \mathbf{\tilde{A}} = \begin{bmatrix} \mathbf{S}^{norm} & \mathbf{R}^{norm} \\
        (\mathbf{R}^{norm})^T & 0 \end{bmatrix}
        """
        num_users = trust_adj.size(0)
        num_items = adj.size(0) - num_users

        # trust_adj (social graph)에서 indices와 values 가져오기
        trust_indices = trust_adj._indices()
        trust_values = trust_adj._values()

        # adj (user-item interaction matrix)에서 indices와 values 가져오기
        adj_indices = adj._indices()
        adj_values = adj._values()

        # User-Item Interaction Matrix의 row index는 user, col index는 item에 해당
        adj_row = adj_indices[0, :]
        adj_col = adj_indices[1, :] 

        # Social Graph와 User-Item Graph의 indices 및 values 결합
        combined_row = t.cat([trust_indices[0, :], adj_row])
        combined_col = t.cat([trust_indices[1, :], adj_col])
        combined_values = t.cat([trust_values, adj_values])

        # (num_users + num_items, num_users + num_items) 크기의 sparse matrix 생성
        combined_size = (num_users + num_items, num_users + num_items)
        combined_adj = t.sparse_coo_tensor(
            indices=t.stack([combined_row, combined_col]),
            values=combined_values,
            size=combined_size,
            device=self.device
        )

        return combined_adj.coalesce()

    def _propagate(self, adj, embeds):
        """전파 함수"""
        # Combined Adj와 입력 임베딩을 사용한 전파
        propagated_embeds = t.sparse.mm(adj, embeds)
        return propagated_embeds
    
    ######################################################
    def _get_trust_influence_mat(self, trust_mat, user_embeds):
        """
        사용자 임베딩에 기반하여 trust graph의 가중치를 조정한 trust influence matrix 생성.

        Args:
            trust_mat (torch.sparse.FloatTensor): Social trust graph (sparse).
            user_embeds (torch.Tensor): 사용자 임베딩 (dense).

        Returns:
            torch.sparse.FloatTensor: Adjusted trust influence matrix (sparse).
        """
        user_norm_embeds = user_embeds / t.norm(user_embeds, p=2, dim=1, keepdim=True)
        # 희소 행렬 비-제로 위치 추출
        indices = trust_mat._indices()
        values = trust_mat._values()
        # 희소 행렬에 해당하는 사용자 임베딩 간 코사인 유사도 계산
        user_i = indices[0]  # source 노드
        user_j = indices[1]  # target 노드
        cosine_sim = (user_norm_embeds[user_i] * user_norm_embeds[user_j]).sum(dim=1)
        user_norm_sim = (1 + cosine_sim) / 2  # 코사인 유사도를 [0, 1]로 정규화
        # influence(u,v) = (1+cos(z_u, z_v))/2 * exp(-|z_u-z_v|^2 / (2*sigma^2))
        if 'sigma' in configs['model']:
            sigma = configs['model']['sigma']
            squared_user_embeds = (user_embeds ** 2).sum(dim=1)
            # RBF kernel 계산 (희소 연산)
            dist_squared = squared_user_embeds[user_i] - 2 * (user_embeds[user_i] * user_embeds[user_j]).sum(dim=1) + squared_user_embeds[user_j]
            kernel_values = t.exp(-dist_squared / (2 * sigma ** 2))
            # RBF kernel과 cosine similarity 결합
            adjusted_values = user_norm_sim * kernel_values
        else:
            adjusted_values = user_norm_sim
    
        # 새로운 희소 행렬 생성
        trust_influence_mat = t.sparse_coo_tensor(indices, adjusted_values, size=trust_mat.shape)
        return trust_influence_mat
    ######################################################
    
    ######################################################
    def _normalize_trust_matrix(self, trust_mat):
        """
        TODO(HYEOKTAE-nim): Sparse trust graph를 대칭 정규화.
        Args:
            trust_mat (torch.sparse.FloatTensor): Trust graph matrix (sparse).

        Returns:
            torch.sparse.FloatTensor: Normalized trust graph (sparse).
        """
        return self._normalize_sparse_matrix(trust_mat)
    ######################################################
    
    ######################################################
    def _socially_aware_normalize_trust_matrix(self, trust_mat):
        """Trust Matrix (torch)에서 socially aware normalize하는 함수"""
        """D^(-1/2) A D^(-1/2) D = D^(-1/2) A D^(1/2)"""
        # 행 합 계산 (degree)
        degree = t.sparse.sum(mat, dim=1).to_dense()
        degree_sqrt = t.pow(degree, 0.5)
        degree_inv_sqrt = t.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0  # 무한대 값 처리

        # 정규화된 행렬 계산: \mathbf{D}^{-1/2} * \mathbf{A} * \mathbf{D}^{-1/2}
        values = mat.values()
        indices = mat.indices()

        row = indices[0]
        col = indices[1]

        # 정규화 값 계산
        norm_values = degree_inv_sqrt[row] * values * degree_sqrt[col]
        normalized_mat = t.sparse_coo_tensor(indices, norm_values, mat.size(), device=self.device)

        return normalized_mat.coalesce()
    ######################################################
    
    ######################################################
    def get_trust_adj(self, trust_mat, adj, embeds):
        """
        Social trust graph와 user-item graph를 결합한 combined adjacency matrix 생성.

        Args:
            trust_mat (torch.sparse.FloatTensor): Social trust graph (sparse).
            adj (torch.sparse.FloatTensor): User-item interaction graph (sparse).
            embeds (torch.Tensor): User + item embeddings (dense).

        Returns:
            torch.sparse.FloatTensor: trust adjacency matrix with influence relaxation (sparse).
        """
        if 'socially_aware_normalization' in configs['model'] and configs['model']['socialyl_aware_normalization']:
            trust_adj = self._socially_aware_normalize_trust_matrix(trust_mat)
        
        else:
            # 1-hop propagation을 통해 user 임베딩 계산
            user_embeds_first_gcn = t.sparse.mm(adj, embeds)[:self.user_num]
            # 사용자 간의 유사도를 반영한 trust influence matrix 생성
            trust_influence_mat = self._get_trust_influence_mat(trust_mat, user_embeds_first_gcn)
            
            if 'pagerank' in configs['model'] and configs['model']['pagerank']:
                # Pagerank 사용 시 adjacency matrix 변경
                trust_adj = self.pagerank_normalized_trust_matrix.to_dense() * trust_influence_mat.to_dense()
                trust_adj += t.eye(self.user_num, device=self.device) # self-loop
                trust_adj = trust_adj.to_sparse()
            else:
                # Trust influence matrix 정규화
                trust_adj = self._normalize_trust_matrix(trust_mat) * trust_influence_mat
        
        return trust_adj
    ######################################################
    
    
    def forward(self):

        #### modified
        u_emb = self.user_embeds
        uu_sim = t.mm(u_emb, u_emb.t())

        threshold_quantile = configs['model']['similarity_threshold']
        self.trust_mat_new = self._reroute_trust_matrix(self.trust_mat, uu_sim, threshold_quantile)

        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        
        if 'combine_gcn' in configs['model'] and configs['model']['combine_gcn']:
            trust_adj = self.get_trust_adj(self.trust_mat_new, self.adj, embeds)
            combined_adj = self._create_combined_adj(self.adj, trust_adj)
            
            for i in range(self.layer_num):
                embeds = self._propagate(combined_adj, embeds_list[-1])
                embeds_list.append(embeds)
            
            embeds = sum(embeds_list)# / len(embeds_list)
        
        else:
            user_embeds = self.user_embeds

            if 'self_gating_unit' in configs['model'] and configs['model']['self_gating_unit']:
                social_user_embeds = self.self_gating_unit_social(user_embeds)
                interaction_user_embeds = self.self_gating_unit_interaction(user_embeds)
            else:
                social_user_embeds = user_embeds
                interaction_user_embeds = user_embeds
            social_embeds = social_user_embeds
            interaction_embeds = t.concat([interaction_user_embeds, self.item_embeds], axis=0)
            social_embeds_list = [social_embeds]
            interaction_embeds_list = [interaction_embeds]
            
            interaction_adj = self.adj
            trust_adj = self.get_trust_adj(self.trust_mat_new, self.adj, embeds)
            
            for i in range(self.layer_num):
                interaction_embeds = self._propagate(interaction_adj, interaction_embeds_list[-1])
                interaction_embeds_list.append(interaction_embeds)
                social_embeds = self._propagate(trust_adj, social_embeds_list[-1])
                social_embeds_list.append(social_embeds)
            
            interaction_embeds = sum(interaction_embeds_list)# / len(interaction_embeds_list)
            social_embeds = sum(social_embeds_list)# / len(social_embeds_list)
            
            if 'alpha' in configs['model']:
                alpha = configs['model']['alpha']
                user_embeds = alpha * interaction_embeds[:self.user_num] + (1 - alpha) * social_embeds
                item_embeds = interaction_embeds[self.user_num:]
            else:
                user_embeds = interaction_embeds[:self.user_num] + social_embeds # exception
                item_embeds = interaction_embeds[self.user_num:]
            
            embeds = t.concat([user_embeds, item_embeds], axis=0)
        
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]

        # if 'combine_gcn' in configs['model'] and configs['model']['combine_gcn']:
        #     trust_adj = self.get_trust_adj(self.trust_mat, self.adj, embeds)
        #     combined_adj = self._create_combined_adj(self.adj, trust_adj)
            
        #     for i in range(self.layer_num):
        #         embeds = self._propagate(combined_adj, embeds_list[-1])
        #         embeds_list.append(embeds)
            
        #     embeds = sum(embeds_list)# / len(embeds_list)
        
        # else:
        #     user_embeds = self.user_embeds

        #     if 'self_gating_unit' in configs['model'] and configs['model']['self_gating_unit']:
        #         social_user_embeds = self.self_gating_unit_social(user_embeds)
        #         interaction_user_embeds = self.self_gating_unit_interaction(user_embeds)
        #     else:
        #         social_user_embeds = user_embeds
        #         interaction_user_embeds = user_embeds
        #     social_embeds = social_user_embeds
        #     interaction_embeds = t.concat([interaction_user_embeds, self.item_embeds], axis=0)
        #     social_embeds_list = [social_embeds]
        #     interaction_embeds_list = [interaction_embeds]
            
        #     interaction_adj = self.adj
        #     trust_adj = self.get_trust_adj(self.trust_mat, self.adj, embeds)
            
        #     for i in range(self.layer_num):
        #         interaction_embeds = self._propagate(interaction_adj, interaction_embeds_list[-1])
        #         interaction_embeds_list.append(interaction_embeds)
        #         social_embeds = self._propagate(trust_adj, social_embeds_list[-1])
        #         social_embeds_list.append(social_embeds)
            
        #     interaction_embeds = sum(interaction_embeds_list)# / len(interaction_embeds_list)
        #     social_embeds = sum(social_embeds_list)# / len(social_embeds_list)
            
        #     if 'alpha' in configs['model']:
        #         alpha = configs['model']['alpha']
        #         user_embeds = alpha * interaction_embeds[:self.user_num] + (1 - alpha) * social_embeds
        #         item_embeds = interaction_embeds[self.user_num:]
        #     else:
        #         user_embeds = interaction_embeds[:self.user_num] + social_embeds # exception
        #         item_embeds = interaction_embeds[self.user_num:]
            
        #     embeds = t.concat([user_embeds, item_embeds], axis=0)
        
        # self.final_embeds = embeds
        # return embeds[:self.user_num], embeds[self.user_num:]
    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        
        
        if 'cl_weight' in configs['model']:
            cl_weight = configs['model']['cl_weight']

            cl_loss_item = cal_infonce_loss(pos_embeds, pos_embeds, item_embeds)
            
            cl_loss = cl_weight * cl_loss_item
            loss += cl_loss
            losses['cl_loss'] = cl_loss
            
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
