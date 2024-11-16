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

class IDEA_MHCN_V4(BaseModel):
    def __init__(self, data_handler):
        super(IDEA_MHCN_V4, self).__init__(data_handler)
        self.data_handler = data_handler
        self._load_configs()
        self._initialize_parameters()
        
        self.is_training = True
    
    def _load_configs(self):
        self.layer_num = configs['model']['layer_num']
        self.reg_weight = configs['model']['reg_weight']
        self.ss_rate = configs['model']['ss_rate']
        
        

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
        self.trn_mat = self.trn_mat.coalesce().to(configs['device'])
        self.trust_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(trust_mat)
        self.trust_mat = self.trust_mat.coalesce().to(configs['device'])
        
        ############################################################
        self.trust_mat = self._make_new_social_mat_v4_4(self.trn_mat, self.trust_mat)
        ############################################################
        
        self._load_data(self.trust_mat, self.trn_mat)
    
    
    
    def _make_new_social_mat_v4_4(self, trn_mat, trust_mat):
        # 1. RR^T 계산 (공통된 아이템 수)
        rr_t = t.sparse.mm(trn_mat, trn_mat.transpose(0, 1)).coalesce()

        # 2. Self-loop 제거 (row_indices == col_indices)
        row_indices, col_indices = rr_t.indices()
        rr_t_values = rr_t.values()

        # Self-loop가 아닌 인덱스만 선택
        non_self_loop_mask = row_indices != col_indices
        row_indices = row_indices[non_self_loop_mask]
        col_indices = col_indices[non_self_loop_mask]
        rr_t_values = rr_t_values[non_self_loop_mask]

        # 3. 새로운 RR^T sparse tensor 생성 (self-loop 제거 후)
        rr_t = t.sparse_coo_tensor(
            t.stack([row_indices, col_indices]),
            rr_t_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 4. 각 노드의 degree 계산 (|B| 계산)
        node_degrees = t.sparse.sum(trn_mat, dim=1).to_dense()

        # 5. Overlap Coefficient 계산 (A → B: |A ∩ B| / |B|)
        overlap_coeff_values = rr_t_values / node_degrees[col_indices].clamp(min=1)

        # 4. trust_mat의 indices로 필터링 (메모리 효율적 방법 사용)
        social_indices = trust_mat.indices().t().tolist()
        social_set = set(map(tuple, social_indices))

        # 5. rr_t의 인덱스를 (row, col) 튜플로 변환하여 필터링
        rr_t_pairs = list(zip(row_indices.tolist(), col_indices.tolist()))
        mask = [pair in social_set for pair in rr_t_pairs]

        # 5. 필터링된 값으로 새로운 tensor 생성
        filtered_row_indices = row_indices[mask]
        filtered_col_indices = col_indices[mask]
        filtered_values = overlap_coeff_values[mask]

        # 6. 새로운 sparse tensor 생성
        updated_social_tensor = t.sparse_coo_tensor(
            t.stack([filtered_row_indices, filtered_col_indices]),
            filtered_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 8. 출력 확인
        print("\nUpdated Social Graph Tensor with Overlap Coefficient:")
        print(updated_social_tensor)

        # 9. 업데이트된 sparse matrix 반환
        return updated_social_tensor
    
    
    def _make_new_social_mat_v4_3(self, trn_mat, trust_mat):
        # 1. RR^T 계산 (공통된 아이템 수)
        rr_t = t.sparse.mm(trn_mat, trn_mat.transpose(0, 1)).coalesce()

        # 2. Self-loop 제거 (row_indices == col_indices)
        row_indices, col_indices = rr_t.indices()
        rr_t_values = rr_t.values()

        # Self-loop가 아닌 인덱스만 선택
        non_self_loop_mask = row_indices != col_indices
        row_indices = row_indices[non_self_loop_mask]
        col_indices = col_indices[non_self_loop_mask]
        rr_t_values = rr_t_values[non_self_loop_mask]

        # 3. 새로운 RR^T sparse tensor 생성 (self-loop 제거 후)
        rr_t = t.sparse_coo_tensor(
            t.stack([row_indices, col_indices]),
            rr_t_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 4. (row, col) 인덱스를 튜플 형식으로 변환
        social_set = set(map(tuple, trust_mat.indices().t().tolist()))
        rr_t_set = set(map(tuple, rr_t.indices().t().tolist()))
        
        # 5. social_set의 개수 계산
        num_social_edges = len(social_set)

        # 6. RR^T value 기준으로 내림차순 정렬
        sorted_indices = t.argsort(rr_t_values, descending=True)
        sorted_row_indices = row_indices[sorted_indices]
        sorted_col_indices = col_indices[sorted_indices]
        sorted_rr_t_values = rr_t_values[sorted_indices]

        # 7. 상위 num_social_edges 개수 선택
        top_row_indices = sorted_row_indices[:num_social_edges]
        top_col_indices = sorted_col_indices[:num_social_edges]

        # 8. 값은 모두 1로 설정
        top_values = t.ones(num_social_edges, device=trust_mat.device)

        # 9. 새로운 sparse tensor 생성
        new_rr_t_tensor = t.sparse_coo_tensor(
            t.stack([top_row_indices, top_col_indices]),
            top_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 10. 출력 확인
        print("\nNew RR^T Tensor (Top sorted by values):")
        print(new_rr_t_tensor)

        # 11. 새로운 sparse matrix 반환
        return new_rr_t_tensor
    
    
    def _make_new_social_mat_v4_2(self, trn_mat, trust_mat):
        # 1. RR^T 계산 (공통된 아이템 수)
        rr_t = t.sparse.mm(trn_mat, trn_mat.transpose(0, 1)).coalesce()

        # 2. Self-loop 제거 (row_indices == col_indices)
        row_indices, col_indices = rr_t.indices()
        rr_t_values = rr_t.values()

        # Self-loop가 아닌 인덱스만 선택
        non_self_loop_mask = row_indices != col_indices
        row_indices = row_indices[non_self_loop_mask]
        col_indices = col_indices[non_self_loop_mask]
        rr_t_values = rr_t_values[non_self_loop_mask]

        # 3. 새로운 RR^T sparse tensor 생성 (self-loop 제거 후)
        rr_t = t.sparse_coo_tensor(
            t.stack([row_indices, col_indices]),
            rr_t_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 4. 각 노드의 연결도(degree) 계산
        node_degrees = t.sparse.sum(trn_mat, dim=1).to_dense()

        # 5. Directed Overlap Coefficient 계산 (A → B: |A ∩ B| / |B|)
        overlap_coefficients = rr_t_values / node_degrees[col_indices]

        # 6. Directed Overlap Coefficient 기준으로 내림차순 정렬
        sorted_indices = t.argsort(overlap_coefficients, descending=True)
        sorted_row_indices = row_indices[sorted_indices]
        sorted_col_indices = col_indices[sorted_indices]
        sorted_rr_t_values = rr_t_values[sorted_indices]

        # 7. social_set의 개수 계산
        social_set = set(map(tuple, trust_mat.indices().t().tolist()))
        num_social_edges = len(social_set)

        # 8. 상위 num_social_edges 개수 선택
        top_row_indices = sorted_row_indices[:num_social_edges]
        top_col_indices = sorted_col_indices[:num_social_edges]

        # 9. 값은 모두 1로 설정
        top_values = t.ones(num_social_edges, device=trust_mat.device)

        # 10. 새로운 sparse tensor 생성
        new_rr_t_tensor = t.sparse_coo_tensor(
            t.stack([top_row_indices, top_col_indices]),
            top_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 11. 출력 확인
        print("\nNew RR^T Tensor (Top sorted by Directed Overlap Coefficient):")
        print(new_rr_t_tensor)

        # 12. 새로운 sparse matrix 반환
        return new_rr_t_tensor
    
    
    def _make_new_social_mat_v4(self, trn_mat, trust_mat):
        # 1. RR^T 계산 (공통된 아이템 수)
        rr_t = t.sparse.mm(trn_mat, trn_mat.transpose(0, 1)).coalesce()

        # 2. Self-loop 제거 (row_indices == col_indices)
        row_indices, col_indices = rr_t.indices()
        rr_t_values = rr_t.values()

        # Self-loop가 아닌 인덱스만 선택
        non_self_loop_mask = row_indices != col_indices
        row_indices = row_indices[non_self_loop_mask]
        col_indices = col_indices[non_self_loop_mask]
        rr_t_values = rr_t_values[non_self_loop_mask]

        # 3. 새로운 RR^T sparse tensor 생성 (self-loop 제거 후)
        rr_t = t.sparse_coo_tensor(
            t.stack([row_indices, col_indices]),
            rr_t_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 4. (row, col) 인덱스를 튜플 형식으로 변환
        social_set = set(map(tuple, trust_mat.indices().t().tolist()))
        rr_t_set = set(map(tuple, rr_t.indices().t().tolist()))

        # 5. 교집합 인덱스 찾기
        intersection_indices = list(social_set & rr_t_set)
        if intersection_indices:
            intersection_indices_tensor = t.tensor(intersection_indices, device=trust_mat.device).t()
            intersection_values = t.ones(intersection_indices_tensor.shape[1], device=trust_mat.device)

            # 새로운 sparse tensor 생성
            intersection_tensor = t.sparse_coo_tensor(
                intersection_indices_tensor,
                intersection_values,
                size=trust_mat.shape,
                device=trust_mat.device
            ).coalesce()
        else:
            print("No overlapping indices found.")
            intersection_tensor = trust_mat.clone()

        # 6. 추가할 인덱스 개수 계산
        num_intersection = len(intersection_indices)
        num_missing = len(social_set) - num_intersection
        print(f"Number of missing indices to refill: {num_missing}")    

        # 6. 교집합 인덱스를 제외한 remaining 인덱스 필터링
        intersection_indices_set = set(intersection_indices)
        remaining_indices_mask = [
            (i, j) not in intersection_indices_set
            for i, j in zip(row_indices.tolist(), col_indices.tolist())
        ]
        remaining_row_indices = row_indices[remaining_indices_mask]
        remaining_col_indices = col_indices[remaining_indices_mask]
        remaining_values = rr_t_values[remaining_indices_mask]

        # 7. remaining tensor 생성
        remaining_tensor = t.sparse_coo_tensor(
            t.stack([remaining_row_indices, remaining_col_indices]),
            remaining_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 9. remaining tensor에서 상위 num_missing 개수 선택 (값 기준 정렬)
        sorted_indices = t.argsort(remaining_values, descending=True)[:num_missing]
        top_row_indices = remaining_row_indices[sorted_indices]
        top_col_indices = remaining_col_indices[sorted_indices]
        top_values = remaining_values[sorted_indices]
        
        # 10. 추가할 sparse tensor 생성
        additional_tensor = t.sparse_coo_tensor(
            t.stack([top_row_indices, top_col_indices]),
            top_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()

        # 11. intersection tensor와 추가된 tensor 합치기
        # updated_social_mat = (intersection_tensor + additional_tensor).coalesce() # lastfm R@10 쭉 오름
        
        # 11. intersection tensor와 추가된 tensor 합치기
        combined_indices = t.cat([intersection_tensor.indices(), additional_tensor.indices()], dim=1)
        combined_values = t.ones(combined_indices.shape[1], device=trust_mat.device)

        # 12. 최종 updated_social_mat 생성 (모든 값을 1로 설정)
        updated_social_mat = t.sparse_coo_tensor(
            combined_indices,
            combined_values,
            size=trust_mat.shape,
            device=trust_mat.device
        ).coalesce()


        # 9. 출력 확인
        print("\nUpdated Social Matrix:")
        print(updated_social_mat)


        # 14. 업데이트된 trust_mat 반환
        self.trust_mat = updated_social_mat
        return updated_social_mat
    
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
        
        
        
        loss = bpr_loss + reg_loss + ss_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'ss_loss': ss_loss}
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
