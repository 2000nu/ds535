
# DS535 Project

This project implements and evaluates SSL-based recommendation models (`idea_denoising`, `idea_rerouting`, and `lightgcn_social`) on datasets like `lastfm` and `yelp`.
The models include features like self-gating units, pagerank-based graph propagation, and hyperparameter tuning.
Our proposing models are `idea_denoising` and `idea_rerouting`.

---

## Setup Instructions

### 1. Install Dependencies

Ensure that all dependencies are installed using `pip`. Run the following command:

```bash
pip install -r requirements.txt
```

---

## Running the Models

To train and test the models, you can use the following commands. Replace the placeholders (`idea_denoising`, `idea_rerouting`, or `lightgcn_social`) and dataset names (`lastfm`, `yelp`) as needed.

### Example Commands

```bash
python main.py --model idea_denoising --dataset lastfm --device cuda --cuda 0
python main.py --model idea_rerouting --dataset lastfm --device cuda --cuda 1
python main.py --model lightgcn_social --dataset yelp --device cuda --cuda 2
```

---

## Configuration

The project supports flexible configuration through `.yml` files located in the `config/modelconf/` directory. Below are the key configuration options:

### General Configuration

- **`optimizer`**: Defines the optimizer type, learning rate (`lr`), and weight decay.
- **`train`**: Controls training parameters like `epoch`, `batch_size`, `loss` type, and early stopping.
- **`test`**: Specifies evaluation metrics (`recall`, `ndcg`, etc.), `k` values for top-K recommendations, and test batch size.
- **`data`**: Defines the dataset type (`social`, `general_cf`, etc.) and dataset name.
- **`model`**: Sets model-specific options like `layer_num`, `embedding_size`, and other hyperparameters.

### Model-Specific Highlights

#### `idea_denoising.yml`
- Implements self-gating units, pagerank, and configurable hyperparameters like `alpha`, `cl_weight`, and `sigma`.
- Supports hyperparameter tuning for `layer_num`, `reg_weight`, and `cl_weight`.

#### `idea_rerouting.yml`
- Focuses on similarity threshold adjustments and routing-based graph updates.
- Configurable hyperparameters include `similarity_threshold` and `alpha`.

#### `lightgcn_social.yml`
- Combines user-item interaction graphs with social trust graphs for LightGCN.
- Tunable parameters include `layer_num` and `reg_weight`.

---

## Enabling Hyperparameter Tuning

To enable grid search for optimal hyperparameters, set the `tune` configuration in the respective `.yml` files:

```yaml
tune:
  enable: true
  hyperparameters: [layer_num, reg_weight]
  layer_num: [2, 3]
  reg_weight: [1.0e-6, 1.0e-7, 1.0e-8]
```

Run the following command to start the tuning process:

```bash
python main.py --model idea_rerouting --dataset yelp --device cuda --cuda 0
```

---

## Reference

This implementation is based on [SSLRec](https://github.com/HKUDS/SSLRec/tree/main). 

Explore the repository for further details on the algorithms and dataset configurations.
