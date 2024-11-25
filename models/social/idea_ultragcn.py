import torch as t
from torch import nn
from config.configurator import configs

class IDEA_ULTRAGCN(nn.Module):
    def __init__(self, data_handler):
        super(IDEA_ULTRAGCN, self).__init__()
        self.data_handler = data_handler
        
        # Load necessary configurations from the data handler
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

        self._load_hyperparameters()
        self._init_parameters()
        
        # Construct item co-occurrence matrix using item-user-user-item paths
        ii_neighbor_mat, ii_constraint_mat = self._get_ii_constraint_mat()
        self.ii_neighbor_mat = ii_neighbor_mat
        self.ii_constraint_mat = ii_constraint_mat
        
        # Compute \Omega to extend UltraGCN to the item-item co-occurrence graph
        constraint_mat = self._prepare_constraint_mat()
        self.constraint_mat = ii_constraint_mat

    def _load_hyperparameters(self):
        # Load hyperparameters
        self.ii_neighbor_num = configs['model']['ii_neighbor_num']
        self.w1 = configs['model']['w1']
        self.w2 = configs['model']['w2']
        self.w3 = configs['model']['w3']
        self.w4 = configs['model']['w4']
        self.gamma = configs['model']['gamma']
        self.lambda_ = configs['model']['lambda']
        self.negative_num = configs['model']['negative_weight']
        self.negative_weight = configs['model']['negative_weight']

    def _init_parameters(self):

        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        trn_mat = self.data_handler.trn_mat
        trust_mat = self.data_handler.trust_mat
        
        self.trn_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(trn_mat)
        self.trn_mat = self.trn_mat.coalesce().to(configs['device'])
        self.trust_mat = self.data_handler._sparse_mx_to_torch_sparse_tensor(trust_mat)
        self.trust_mat = self.trust_mat.coalesce().to(configs['device'])
        

    def _get_ii_constraint_mat(self, ii_diagonal_zero=False):
        """
        Compute the item-item constraint matrix (Ω) for the item co-occurrence graph.
        
        Args:
            train_mat (scipy.sparse.csr_matrix): User-item interaction matrix.
            num_neighbors (int): Number of neighbors to consider for each item.
            ii_diagonal_zero (bool): If True, sets diagonal of item-item co-occurrence matrix to zero.

        Returns:
            res_mat (torch.Tensor): Tensor of shape [num_items, num_neighbors] containing neighbor indices.
            res_sim_mat (torch.Tensor): Tensor of shape [num_items, num_neighbors] containing similarity scores.
        """
        device = configs['device']
        trn_mat = self.trn_mat
        num_neighbors = self.ii_neighbor_num
        
        

        print('Computing \\Omega for the item-item graph...')
        
        # TODO: including social part in here
        # Compute item-item co-occurrence matrix: A = train_mat.T * train_mat
        A = t.sparse.mm(trn_mat.T, trn_mat).coalesce()
        
        # trust_mat = self.trust_mat
        # rt_s = t.sparse.mm(trn_mat.T, trust_mat).coalesce()
        # A = t.sparse.mm(rt_s, trn_mat).coalesce()

        n_items = A.shape[0]

        # Initialize result tensors
        ii_neighbor_mat = t.zeros((n_items, num_neighbors), dtype=t.long, device=device)
        ii_constraint_mat = t.zeros((n_items, num_neighbors), dtype=t.float32, device=device)

        # Compute degree vectors
        items_D = t.sparse.sum(A, dim=0).to_dense() + 1e-8
        users_D = t.sparse.sum(trn_mat, dim=1).to_dense() + 1e-8

        # Calculate scaling factors (beta_uD and beta_iD)
        beta_uD = (t.sqrt(users_D + 1) / users_D).reshape(-1, 1).to(device)
        beta_iD = (1 / t.sqrt(items_D + 1)).reshape(-1).to(device)

        # Compute constraint matrix using element-wise multiplication
        all_ii_constraint_mat = beta_uD @ beta_iD.unsqueeze(0)
        
        row_idx, col_idx = A.indices()
        values = A.values()
        
        # Iterate over each item to find top-k neighbors
        for i in range(n_items):
            # Extract non-zero values and indices for the current row            
            print(i)
            row_indices = col_idx[row_idx == i]
            row_values = values[row_idx == i]

            # Compute similarity scores for item i
            row_sims = all_ii_constraint_mat[i, row_indices] * row_values
            
            
            if len(row_sims) < num_neighbors:
                # Fill the remaining with zeros (padding)
                top_sims = t.zeros(num_neighbors, device=device)
                top_idxs = t.zeros(num_neighbors, dtype=t.long, device=device)
                if len(row_sims) > 0:
                    # Get the top available similarities
                    top_sims[:len(row_sims)], top_idxs[:len(row_sims)] = t.topk(row_sims, len(row_sims), largest=True)
                    top_idxs[:len(row_sims)] = row_indices[top_idxs[:len(row_sims)]]
            else:
                # Get the top-k similarities
                top_sims, top_idxs = t.topk(row_sims, num_neighbors, largest=True)
            
            # top_sims, top_idxs = t.topk(row_sims, num_neighbors, largest=True)

            # Store the indices and similarity scores
            ii_neighbor_mat[i] = row_idx[top_idxs]
            ii_constraint_mat[i] = top_sims

            # Log progress every 15,000 items
            if i % 15000 == 0:
                print(f"Processed item {i}/{n_items}")

        print("Computation \\Omega completed!")
        return ii_neighbor_mat, ii_constraint_mat


    def _prepare_constraint_mat(self):
        """
        Compute the constraint matrix for GraphMF using the given user-item interaction matrix.

        Args:
            train_mat (torch.sparse.Tensor): User-item interaction matrix in sparse format.

        Returns:
            constraint_mat (dict): Dictionary containing 'beta_uD' and 'beta_iD' for omega weight calculations.
        """
        train_mat = self.trn_mat
        device = train_mat.device

        # Compute the degree of users and items
        user_degrees = torch.sparse.sum(train_mat, dim=1).to_dense()  # Shape: [n_user]
        item_degrees = torch.sparse.sum(train_mat, dim=0).to_dense()  # Shape: [m_item]

        # Calculate scaling factors (beta_uD and beta_iD) for the constraint matrix
        beta_uD = (torch.sqrt(user_degrees + 1) / (user_degrees + 1e-8)).reshape(-1, 1)
        beta_iD = (1 / torch.sqrt(item_degrees + 1)).reshape(-1)

        # Construct the constraint matrix as a dictionary
        constraint_mat = {
            "beta_uD": beta_uD.squeeze().to(device),
            "beta_iD": beta_iD.squeeze().to(device)
        }

        return constraint_mat


    def forward(self, users, items):
        # Retrieve user and item embeddings
        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding(items)

        # Compute predictions using dot product between user and item embeddings
        predictions = (user_embeds * item_embeds).sum(dim=1)
        return predictions

    def cal_loss(self, batch_data):
        users, pos_items, neg_items = batch_data

        # Get omega weights
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        # Calculate loss components
        loss_L = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss_I = self.cal_loss_I(users, pos_items)
        norm_loss = self.norm_loss()

        # Total loss
        loss = loss_L + self.lambda_reg * loss_I + self.gamma * norm_loss
        losses = {'loss_L': loss_L.item(), 'loss_I': loss_I.item(), 'norm_loss': norm_loss.item()}

        return loss, losses

    def get_omegas(self, users, pos_items, neg_items):
        device = configs['device']

        pos_weight = (self.constraint_mat['beta_uD'][users] * self.constraint_mat['beta_iD'][pos_items]).to(device)
        pos_weight = self.w1 + self.w2 * pos_weight if self.w2 > 0 else self.w1

        neg_weight = (t.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)) *
                      self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
        neg_weight = self.w3 + self.w4 * neg_weight if self.w4 > 0 else self.w3

        return t.cat([pos_weight, neg_weight])

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.user_embedding.weight.device

        user_embeds = self.user_embedding(users)
        pos_embeds = self.item_embedding(pos_items)
        neg_embeds = self.item_embedding(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)
        neg_scores = (user_embeds.unsqueeze(1) * neg_embeds).sum(dim=-1)

        pos_labels = t.ones_like(pos_scores, device=device)
        neg_labels = t.zeros_like(neg_scores, device=device)

        pos_loss = nn.functional.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)], reduction='none')
        neg_loss = nn.functional.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight=omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim=-1)

        return (pos_loss + neg_loss * self.negative_weight).sum()

    def cal_loss_I(self, users, pos_items):
        device = configs['device']
        user_embeds = self.user_embedding(users).unsqueeze(1)
        neighbor_embeds = self.item_embedding(self.ii_neighbor_mat[pos_items].to(device))
        sim_scores = self.ii_constraint_mat[pos_items].to(device)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
        return loss.sum()

    def norm_loss(self):
        loss = sum((param ** 2).sum() for param in self.parameters())
        return loss / 2

    def full_predict(self, batch_data):
        batch_users, train_mask = batch_data
        user_embeds = self.user_embedding(batch_users)
        all_item_embeds = self.item_embedding.weight

        full_preds = t.matmul(user_embeds, all_item_embeds.T)
        return self._mask_predict(full_preds, train_mask)

    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask
