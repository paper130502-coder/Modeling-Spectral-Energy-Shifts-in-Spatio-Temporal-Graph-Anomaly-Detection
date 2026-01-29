# EGNN: Energy-based Graph Neural Networks for Anomaly Detection

This repository contains the official implementation of **Energy-based Graph Neural Networks (EGNN)** for graph-based fraud detection and time-series anomaly detection.

## Installation

```bash
# Clone the repository
git clone https://github.com/paper130502-coder/Modeling-Spectral-Energy-Shifts-in-Spatio-Temporal-Graph-Anomaly-Detection.git


# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=1.10.0
dgl>=0.9.0
numpy
pandas
scikit-learn
```

## Usage

### Graph-based Fraud Detection

```bash
# Train on Amazon dataset
python E_train.py --dataset amazon --epochs 300 --runs 10 --gate_hidden_dim 32 --gate_num_layers 4 

# Train on Yelp dataset
python E_train.py --dataset yelp --hidden_dim 128 --gate_hidden_dim 8  --gate_num_layers 2 --attention_reduction 8
# Train on TFinance
python E_train.py --dataset tfinance --runs 10

# Train on TSocial (large dataset)
python E_train.py --dataset tsocial --runs 10
```

### Time-Series Anomaly Detection

```bash
# Train on MSL dataset
python f_train2.py --dataset msl --epochs 150 --runs 10 --train_ratio 0.01 --slide_win 64 --lr 0.005

# Train on SWaT dataset
python f_train2.py --dataset swat --epochs 80 --runs 10 --train_ratio 0.01 --slide_win 64 --lr 0.005

# Train on WADI dataset
python f_train2.py --dataset wadi --epochs 80 --runs 10 --train_ratio 0.01 --slide_win 64 --lr 0.005
# Preprocess datasets (creates cache for faster loading)
python dataloader_semi.py --preprocess swat wadi --window 15
```

## Command Line Arguments

### E_train.py (Graph-based)

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | yelp | Dataset: amazon, yelp, tfinance, tsocial |
| `--train_ratio` | 0.4 | Training set ratio |
| `--val_ratio` | 0.2 | Validation set ratio |
| `--hidden_dim` | 128 | Hidden layer dimension |
| `--dropout` | 0.5 | Dropout rate |
| `--attention_reduction` | 10 | Attention reduction ratio |
| `--gate_type` | per_feature | Gate type: per_node, per_feature |
| `--gate_hidden_dim` | 16 | Gate MLP hidden dimension |
| `--gate_num_layers` | 2 | Gate MLP layers (2, 3, or 4) |
| `--epochs` | 300 | Training epochs |
| `--lr` | 0.01 | Learning rate |
| `--weight_decay` | 5e-4 | Weight decay |
| `--seed` | 47 | Random seed |
| `--runs` | 10 | Number of runs |
| `--undirected` | False | Use undirected graph |

### f_train2.py (Time-Series)

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | msl | Dataset: msl, swat, wadi |
| `--slide_win` | 15 | Sliding window size |
| `--slide_stride` | 1 | Sliding window stride |
| `--hidden_dim` | 128 | Hidden layer dimension |
| `--dropout` | 0.5 | Dropout rate |
| `--gate_type` | per_feature | Gate type: per_node, per_feature |
| `--gate_hidden_dim` | 16 | Gate MLP hidden dimension |
| `--gate_num_layers` | 2 | Gate MLP layers |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 0.005 | Learning rate |
| `--patience` | 20 | LR scheduler patience |
| `--runs` | 1 | Number of runs |

## Evaluation Metrics

All experiments report comprehensive metrics:

### Classification Metrics
- **Macro F1**: Macro-averaged F1 score
- **Recall**: True positive rate
- **Precision**: Positive predictive value

### Ranking Metrics
- **AUROC**: Area Under ROC Curve
- **AUPRC**: Area Under Precision-Recall Curve
- **RecK**: Recall at K (K = number of anomalies)

## Project Structure

```
egnn_icml/
├── README.md
├── E_train.py              # Main training for graph-based fraud detection
├── f_train2.py             # Training for time-series anomaly detection
├── f_model2.py             # TemporalGatedEnergySAGE model
├── fast_e.py               # Fast local 1-hop energy computation
├── dataloader_semi.py      # Data loader for time-series datasets
├── get_amazon.py           # Amazon dataset loader
├── get_yelp.py             # Yelp dataset loader
├── get_tfinance.py         # TFinance dataset loader
├── get_tsocial.py          # TSocial dataset loader
└── data/                   # Dataset directory
    ├── amazon/
    ├── yelp/
    ├── msl/
    ├── swat/
    └── wadi/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DGL team for the graph neural network library
- GADBench for fraud detection benchmarks
- TGAD for time-series anomaly detection datasets
