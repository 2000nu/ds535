import pandas as pd
import numpy as np
import pickle
from scipy.sparse import coo_matrix

# 파일 경로 설정
file_paths = {
    "train": "train_set.txt",
    "test": "test_set.txt",
    "trust": "trust.txt"
}

output_file_paths = {
    "trn_mat": "trn_mat.pkl",
    "tst_mat": "tst_mat.pkl",
    "trust_mat": "trust_mat.pkl"
}

# 1. 데이터 읽기
train_set = pd.read_csv(file_paths["train"])
test_set = pd.read_csv(file_paths["test"])
trust_set = pd.read_csv(file_paths["trust"])

# 2. 유저와 아이템의 최대 인덱스 계산
n_user = max(train_set["user"].max(), test_set["user"].max(), trust_set["user"].max(), trust_set["friend"].max()) + 1
m_item = max(train_set["item"].max(), test_set["item"].max()) + 1

# 3. Train matrix (trn_mat)
train_rows = train_set["user"].values
train_cols = train_set["item"].values
train_data = np.ones(len(train_rows))
trn_mat = coo_matrix((train_data, (train_rows, train_cols)), shape=(n_user, m_item))

# 4. Test matrix (tst_mat)
test_rows = test_set["user"].values
test_cols = test_set["item"].values
test_data = np.ones(len(test_rows))
tst_mat = coo_matrix((test_data, (test_rows, test_cols)), shape=(n_user, m_item))

# 5. Trust matrix (trust_mat)
trust_rows = trust_set["user"].values
trust_cols = trust_set["friend"].values
trust_data = np.ones(len(trust_rows))
trust_mat = coo_matrix((trust_data, (trust_rows, trust_cols)), shape=(n_user, n_user))

# 6. .pkl 파일로 저장
with open(output_file_paths["trn_mat"], 'wb') as f:
    pickle.dump(trn_mat, f)
with open(output_file_paths["tst_mat"], 'wb') as f:
    pickle.dump(tst_mat, f)
with open(output_file_paths["trust_mat"], 'wb') as f:
    pickle.dump(trust_mat, f)

print("Saved trn_mat.pkl, tst_mat.pkl, trust_mat.pkl with original indices.")
