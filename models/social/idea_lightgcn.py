import torch as t
from torch import nn
from models.aug_utils import EdgeDrop
from models.base_model import BaseModel
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class IDEA_LIGHTGCN(BaseModel):
    def __init__(self, data_handler):
        super(IDEA_LIGHTGCN, self).__init__(data_handler)

        self.device = configs['device']
        
        # Data handler에서 trn_mat과 trust_mat 가져오기
        self.trn_mat = self._coo_to_sparse_tensor(data_handler.trn_mat)
        self.trust_mat = self._coo_to_sparse_tensor(data_handler.trust_mat)

        # COO Matrix -> Sparse Tensor 변환 및 정규화
        A = self._create_adj_matrix(self.trn_mat)
        self.adj = self._normalize_sparse_matrix(A)

        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        self.keep_rate = configs['model']['keep_rate']
        

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        self.edge_dropper = EdgeDrop()
        self.is_training = True
        self.final_embeds = None
    
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
        # Item index를 shift하여 user 다음에 위치하도록 함
        adj_row = adj_indices[0, :]
        adj_col = adj_indices[1, :] + num_users  # item index shift

        # Social Graph와 User-Item Graph의 indices 및 values 결합
        combined_row = t.cat([trust_indices[0, :], adj_row, adj_col])
        combined_col = t.cat([trust_indices[1, :], adj_col, adj_row])
        combined_values = t.cat([trust_values, adj_values, adj_values])

        # (num_users + num_items, num_users + num_items) 크기의 sparse matrix 생성
        combined_size = (num_users + num_items, num_users + num_items)
        combined_adj = t.sparse_coo_tensor(
            indices=t.stack([combined_row, combined_col]),
            values=combined_values,
            size=combined_size,
            device=self.device
        )

        return combined_adj.coalesce()

    def _propagate(self, combined_adj, embeds):
        """전파 함수"""
        # Combined Adj와 입력 임베딩을 사용한 전파
        propagated_embeds = t.sparse.mm(combined_adj, embeds)
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
        user_embeds = user_embeds / t.norm(user_embeds, p=2, dim=1, keepdim=True)
        user_cosine_sim = t.matmul(user_embeds, user_embeds.T)
        user_norm_sim = (1 + user_cosine_sim) / 2
        trust_influence_mat = trust_mat * user_norm_sim
        return trust_influence_mat
    ######################################################
    
    ######################################################
    def _normalize_trust_matrix(self, trust_mat):
        """
        TODO: Sparse trust graph를 대칭 정규화.
        Args:
            trust_mat (torch.sparse.FloatTensor): Trust graph matrix (sparse).

        Returns:
            torch.sparse.FloatTensor: Normalized trust graph (sparse).
        """
        return self._normalize_sparse_matrix(trust_mat)
    ######################################################
    
    ######################################################
    def get_combined_adj(self, trust_mat, adj, embeds):
        """
        Social trust graph와 user-item graph를 결합한 combined adjacency matrix 생성.

        Args:
            trust_mat (torch.sparse.FloatTensor): Social trust graph (sparse).
            adj (torch.sparse.FloatTensor): User-item interaction graph (sparse).
            embeds (torch.Tensor): User + item embeddings (dense).

        Returns:
            torch.sparse.FloatTensor: Combined adjacency matrix (sparse).
        """
        
        # 1-hop propagation을 통해 user 임베딩 계산
        user_embeds_first_gcn = t.sparse.mm(adj, embeds)[:self.user_num]
        # 사용자 간의 유사도를 반영한 trust influence matrix 생성
        trust_influence_mat = self._get_trust_influence_mat(trust_mat, user_embeds_first_gcn)
        # Trust influence matrix 정규화
        trust_adj = self._normalize_trust_matrix(trust_influence_mat)
        # Combined adjacency matrix 생성
        combined_adj = self._create_combined_adj(adj, trust_adj)
        
        return combined_adj
    ######################################################
    
    def forward(self, keep_rate):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0).to(self.device)
        embeds_list = [embeds]
        
        ######################################################
        combined_adj = self.get_combined_adj(self.trust_mat, self.adj, embeds)
        ######################################################
        
        for i in range(self.layer_num):
            embeds = self._propagate(combined_adj, embeds_list[-1])
            embeds_list.append(embeds)
        
        embeds = sum(embeds_list)# / len(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]
    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds