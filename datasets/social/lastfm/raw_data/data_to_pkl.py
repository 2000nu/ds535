from scipy.sparse import coo_matrix, vstack
import numpy as np
import pandas as pd
import pickle
from implicit.evaluation import train_test_split

file_paths = {
    "interactions": "lastfm.inter",
    "network": "lastfm.net",
    "items": "lastfm.item"
}

# Set a random seed for reproducibility
RANDOM_SEED = 42

interactions_df = pd.read_csv(file_paths["interactions"], sep="\t")
network_df = pd.read_csv(file_paths["network"], sep="\t")
items_df = pd.read_csv(file_paths["items"], sep="\t")

# Extract unique users and items from interaction data to define matrix shapes.
users = interactions_df["user_id:token"].astype(int).unique()
items = interactions_df["artist_id:token"].astype(int).unique()
user_count, item_count = len(users), len(items)

# Mapping users and items to matrix indices for sparse representation.
user_to_idx = {user: idx for idx, user in enumerate(users)}
item_to_idx = {item: idx for idx, item in enumerate(items)}

# Map interactions to matrix indices
interaction_rows = interactions_df["user_id:token"].map(user_to_idx).values
interaction_cols = interactions_df["artist_id:token"].map(item_to_idx).values
interaction_data = np.ones(len(interaction_rows))  # Binary interactions

# Create the complete interaction matrix
interaction_matrix = coo_matrix((interaction_data, (interaction_rows, interaction_cols)), shape=(user_count, item_count))

# Apply implicit's train_test_split
# trn_mat, tst_mat = train_test_split(interaction_matrix, train_percentage=0.8)

train_matrices = []
test_matrices = []

# Custom train-test split to ensure all users have data in the training set
for user_id in range(user_count):
    user_interactions = interaction_matrix.getrow(user_id)
    if user_interactions.nnz <= 1:
        # If user has only one interaction, put it all in training
        train_matrices.append(user_interactions)
    else:
        # Split interactions for the user
        user_train, user_test = train_test_split(user_interactions, train_percentage=0.8, random_state=RANDOM_SEED)
        train_matrices.append(user_train)
        test_matrices.append(user_test)

# Stack all user rows back into the full train and test matrices
trn_mat = vstack(train_matrices).tocoo()
tst_mat = vstack(test_matrices).tocoo() if test_matrices else coo_matrix((user_count, item_count))



# Social relationship (trust) matrix.
trust_row = network_df["source_id:token"].map(user_to_idx).values
trust_col = network_df["target_id:token"].map(user_to_idx).values
trust_data = np.ones(len(trust_row))  # Binary social relationship
trust_mat = coo_matrix((trust_data, (trust_row, trust_col)), shape=(user_count, user_count))


# Save each matrix to the respective pkl file.
with open('../trn_mat.pkl', 'wb') as f:
    pickle.dump(trn_mat, f)
with open('../tst_mat.pkl', 'wb') as f:
    pickle.dump(tst_mat, f)
with open('../trust_mat.pkl', 'wb') as f:
    pickle.dump(trust_mat, f)

# Mapping된 데이터를 저장할 파일 경로
output_file_paths = {
    "interactions": "lastfm_idx.inter",
    "network": "lastfm_idx.net"
}

# 1. Interactions 데이터 변환 및 저장
interactions_df["user_idx"] = interactions_df["user_id:token"].map(user_to_idx)
interactions_df["item_idx"] = interactions_df["artist_id:token"].map(item_to_idx)
mapped_interactions_df = interactions_df[["user_idx", "item_idx"]]

mapped_interactions_df.to_csv(output_file_paths["interactions"], sep="\t", index=False, header=False)
print(f"Mapped interactions data saved to {output_file_paths['interactions']}")

# 2. Network 데이터 변환 및 저장
network_df["source_idx"] = network_df["source_id:token"].map(user_to_idx)
network_df["target_idx"] = network_df["target_id:token"].map(user_to_idx)
mapped_network_df = network_df[["source_idx", "target_idx"]]

mapped_network_df.to_csv(output_file_paths["network"], sep="\t", index=False, header=False)
print(f"Mapped network data saved to {output_file_paths['network']}")
