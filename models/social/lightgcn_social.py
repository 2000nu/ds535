import torch as t
from torch import nn
from models.base_model import BaseModel
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

def is_symmetric(matrix, tol=1e-8):
    """
    Check if a sparse matrix is symmetric.

    Args:
        matrix (torch.sparse_coo_tensor): The sparse tensor to check.
        tol (float): Tolerance for comparing floating point values.

    Returns:
        bool: True if symmetric, False otherwise.
    """
    # Ensure the matrix is coalesced
    matrix = matrix.coalesce()
    matrix_t = matrix.transpose(0, 1).coalesce()

    # Get indices and values
    indices = matrix.indices()
    values = matrix.values()
    t_indices = matrix_t.indices()
    t_values = matrix_t.values()

    # Create composite keys for sorting
    N = matrix.size(0)  # Assuming square matrix
    orig_keys = indices[0] * N + indices[1]
    trans_keys = t_indices[0] * N + t_indices[1]

    # Sort the entries
    orig_sort_order = t.argsort(orig_keys)
    trans_sort_order = t.argsort(trans_keys)

    # Apply the sorting order
    orig_indices_sorted = indices[:, orig_sort_order]
    orig_values_sorted = values[orig_sort_order]
    trans_indices_sorted = t_indices[:, trans_sort_order]
    trans_values_sorted = t_values[trans_sort_order]

    # Compare the sorted indices
    indices_equal = t.equal(orig_indices_sorted, trans_indices_sorted)

    # Compare the sorted values within a tolerance
    values_close = t.allclose(orig_values_sorted, trans_values_sorted, atol=tol)

    return indices_equal and values_close

class LIGHTGCN_SOCIAL(BaseModel):
    def __init__(self, data_handler):
        super(LIGHTGCN_SOCIAL, self).__init__(data_handler)

        self.device = configs['device']

        # Data handler에서 trn_mat과 trust_mat 가져오기
        self.trn_mat = self._coo_to_sparse_tensor(data_handler.trn_mat)
        self.trust_mat = self._coo_to_sparse_tensor(data_handler.trust_mat)
        # Determine if the social matrix is symmetric
        is_symmetric_social = is_symmetric(self.trust_mat)
        print(f"The trust matrix is {'symmetric' if is_symmetric_social else 'asymmetric'}.")

        # COO Matrix -> Sparse Tensor 변환 및 정규화
        A = self._create_adj_matrix(self.trn_mat)
        self.adj = self._normalize_sparse_matrix(A, symmetric=True)  # Interaction matrix is usually symmetric
        self.trust_adj = self._normalize_sparse_matrix(self.trust_mat, symmetric=is_symmetric_social)
        self.combined_adj = self._create_combined_adj(self.adj, self.trust_adj)

        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        self.is_training = True
        self.final_embeds = None

    
    def _coo_to_sparse_tensor(self, coo_mat):
        """COO matrix를 torch sparse tensor로 변환"""
        coo_mat = coo_mat.tocoo()
        indices = t.tensor([coo_mat.row, coo_mat.col], dtype=t.long)
        values = t.tensor(coo_mat.data, dtype=t.float32)
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

    def _normalize_sparse_matrix(self, mat, symmetric=True):
        """Normalize a sparse matrix (torch) using symmetric or row normalization."""
        # Calculate degree (sum over rows)
        degree = t.sparse.sum(mat, dim=1).to_dense()
        degree[degree == 0] = 1e-8  # Avoid division by zero

        if symmetric:
            degree_inv_sqrt = t.pow(degree, -0.5)
            degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0  # Handle infinite values

            # Symmetric normalization: D^{-1/2} * A * D^{-1/2}
            values = mat.values()
            indices = mat.indices()
            row, col = indices[0], indices[1]
            norm_values = degree_inv_sqrt[row] * values * degree_inv_sqrt[col]
        else:
            degree_inv = 1.0 / degree
            degree_inv[degree_inv == float('inf')] = 0.0  # Handle infinite values

            # Row normalization: D^{-1} * A
            values = mat.values()
            indices = mat.indices()
            row = indices[0]
            norm_values = degree_inv[row] * values

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

        # adj (user+item interaction matrix)에서 indices와 values 가져오기
        adj_indices = adj._indices()
        adj_values = adj._values()

        # Social Graph와 User-Item Graph의 indices 및 values 결합
        combined_row = t.cat([trust_indices[0, :], adj_indices[0, :]])
        combined_col = t.cat([trust_indices[1, :], adj_indices[1, :]])
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

    def _propagate(self, combined_adj, embeds):
        """전파 함수"""
        # Combined Adj와 입력 임베딩을 사용한 전파
        propagated_embeds = t.sparse.mm(combined_adj, embeds).to(self.device)
        return propagated_embeds
    
    def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0).to(self.device)
        embeds_list = [embeds]

        combined_adj = self.combined_adj
        for i in range(self.layer_num):
            embeds = self._propagate(combined_adj, embeds_list[-1])
            embeds_list.append(embeds)
        
        embeds = sum(embeds_list)# / len(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]
    
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