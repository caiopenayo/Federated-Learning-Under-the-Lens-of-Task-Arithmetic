# Federated Learning Under the Lens of Task Arithmetic

This repository contains the implementation and experiments for the project "Federated Learning Under the Lens of Task Arithmetic", which investigates how task arithmetic principles can help mitigate client drift in federated learning under statistical heterogeneity.

## Authors

- Caio V. M. Penayo (Matr. 350183)
- Tiago G. Felice (Matr. 350158)
- Guilherme Turina (Matr. 350487)
- Imane EL-KAROUMI (Matr. 351002)

## Overview

Federated Learning (FL) enables collaborative training across decentralized data sources while preserving privacy, but faces significant challenges under statistical heterogeneity. This project reinterprets federated learning through the lens of **task arithmetic**, treating client updates as task vectors whose aggregation induces interference when data distributions differ.

### Key Contributions

- Reinterpretation of federated learning as distributed task arithmetic
- Strong centralized baseline on CIFAR-100 using pretrained DINO ViT-S/16 (88.36% test accuracy)
- Systematic evaluation of FedAvg under varying degrees of heterogeneity
- Investigation of sensitivity-aware sparse fine-tuning to mitigate client drift
- Comparison of alternative masking strategies (magnitude-based, random, sensitivity-based)

## Project Structure

```
.
├── centralized_baseline.ipynb          # Centralized training experiments
├── Federated_iid_.ipynb               # FedAvg on IID data
├── federated_learing.ipynb            # General federated learning setup
├── results_fixed_non_iid.ipynb        # FedAvg on non-IID data
├── TaskArithmetic_centralized.ipynb   # Task arithmetic in centralized setting
├── TaskArithmetic_federated.ipynb     # Task arithmetic in federated setting
├── projeto_colab.ipynb                # Main project notebook
├── project.ipynb                      # Code structure and utilities
├── models/                            # Model definitions
│   └── vit_dino.py                   # DINO ViT implementation
├── data/                              # Data loading and partitioning
│   ├── datasets.py                   # CIFAR-100 dataset utilities
│   └── partition.py                  # IID and non-IID partitioning
├── fl/                                # Federated learning utilities
│   └── dataloaders.py                # Federated dataloader construction
├── train/                             # Training utilities
│   ├── trainer.py                    # Training loop
│   └── eval.py                       # Evaluation functions
└── utils/                             # Utility functions
    └── plots.py                      # Plotting functions
```

## Requirements

The project was developed using Google Colab with the free tier. The main dependencies are:

```bash
torch
torchvision
timm
tqdm
matplotlib
numpy
```

### Installation

Install all required packages:

```bash
pip install torch torchvision timm tqdm matplotlib numpy
```

## Datasets

The project uses **CIFAR-100** dataset, which is automatically downloaded via `torchvision`. The dataset is partitioned into:

- Training set: 90% of original training data
- Validation set: 10% of original training data
- Test set: standard CIFAR-100 test set

Images are resized to **160×160** pixels (or 224×224 in some experiments) to reduce computational requirements.

## Running the Experiments

### 1. Centralized Baseline

To establish the centralized baseline performance:

```python
# Open centralized_baseline.ipynb
# This notebook trains a DINO ViT-S/16 model on CIFAR-100
# Performs hyperparameter search over learning rates and weight decay
# Best configuration: LR=0.01, WD=1e-4, Cosine scheduler, 25 epochs
# Expected result: ~88.36% test accuracy
```

Key steps:
- Loads CIFAR-100 with 90/10 train/validation split
- Tests different schedulers (cosine, step, multistep, constant)
- Performs grid search over learning rates and weight decay
- Trains best configuration for 25 epochs

### 2. FedAvg on IID Data

To run FedAvg under IID data distribution:

```python
# Open Federated_iid_.ipynb
# Configuration: K=100 clients, C=0.1 participation rate, J=4 local steps
# Data: IID partitioning (uniform distribution across clients)
# Expected result: ~84.75% test accuracy after 440 rounds
```

Parameters:
- `K=100`: Total number of clients
- `C=0.1`: Client participation fraction (10 clients per round)
- `J=4`: Number of local SGD steps per round
- Batch size: 64

### 3. FedAvg on Non-IID Data

To evaluate FedAvg under statistical heterogeneity:

```python
# Open results_fixed_non_iid.ipynb
# Tests multiple configurations:
# - Nc ∈ {1, 5, 10, 50} (number of classes per client)
# - J ∈ {4, 8, 16} (local steps)
# Shows dramatic performance degradation under extreme heterogeneity
```

Expected results:
- `Nc=50`: ~86% accuracy (near-IID)
- `Nc=10`: Moderate degradation
- `Nc=5`: ~16% accuracy
- `Nc=1`: ~7% accuracy (extreme heterogeneity)

### 4. Task Arithmetic - Centralized

To test sensitivity-aware sparse fine-tuning in centralized setting:

```python
# Open TaskArithmetic_centralized.ipynb
# Implements:
# 1. Fisher Information Matrix computation for sensitivity scores
# 2. Gradient mask calibration (identifies least-sensitive parameters)
# 3. SparseSGDM optimizer (updates only masked parameters)
```

Key parameters:
- **Trainable fraction (tf)**: Proportion of parameters to update (0.1, 0.2)
- **Calibration rounds (cr)**: Number of rounds to calibrate mask (1, 3, 5)

Results show that proper calibration is critical:
- `tf=0.2, cr=5`: ~83% accuracy
- `tf=0.1, cr=5`: ~78% accuracy

### 5. Task Arithmetic - Federated

To apply task arithmetic to federated learning:

```python
# Open TaskArithmetic_federated.ipynb
# Combines sensitivity-aware masking with FedAvg
# Tests under both IID and non-IID conditions
```

Key findings:
- Works well under IID and mild heterogeneity (Nc=50)
- Performance degrades under extreme heterogeneity (Nc=1)
- Sensitivity-aware masking outperforms magnitude-based alternatives

### 6. Extension: Alternative Masking Rules

The project investigates different parameter selection strategies:

1. **Least sensitive** (default): Select parameters with lowest sensitivity scores
2. **Most sensitive**: Select parameters with highest sensitivity scores
3. **Lowest magnitude**: Select parameters with smallest absolute values
4. **Highest magnitude**: Select parameters with largest absolute values
5. **Random**: Random parameter selection

Results confirm that sensitivity-aware selection is superior to magnitude-based heuristics.

## Key Hyperparameters

### Centralized Training
- Learning rate: 0.01
- Weight decay: 1×10⁻⁴
- Optimizer: SGD with momentum 0.9
- Scheduler: Cosine annealing
- Epochs: 25
- Batch size: 64

### Federated Learning
- Number of clients (K): 100
- Client participation (C): 0.1
- Local steps (J): 4, 8, or 16
- Communication rounds: ~440 (to match centralized compute)
- Aggregation: FedAvg (weighted by dataset size)

### Task Arithmetic
- Trainable fraction: 0.1 or 0.2
- Calibration rounds: 1, 3, or 5
- Masking strategy: Least sensitive (sensitivity-aware)

## Reproducing Results

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd federated-task-arithmetic
```

2. **Open in Google Colab**
   - Upload the notebooks to Google Colab
   - Enable GPU runtime: Runtime → Change runtime type → GPU

3. **Run experiments in order**
   - Start with `centralized_baseline.ipynb`
   - Then `Federated_iid_.ipynb`
   - Follow with `results_fixed_non_iid.ipynb`
   - Finally run task arithmetic notebooks

### Expected Runtime

- Centralized baseline: ~2-3 hours
- FedAvg IID: ~4-6 hours
- FedAvg non-IID (all configurations): ~8-12 hours
- Task arithmetic experiments: ~6-10 hours

**Note**: Google Colab may interrupt long-running sessions. Implement checkpointing to resume experiments.

## Main Results

### Performance Summary

| Setting | Configuration | Test Accuracy |
|---------|---------------|---------------|
| Centralized | Full training | 88.36% |
| FedAvg IID | K=100, C=0.1, J=4 | 84.75% |
| FedAvg non-IID | Nc=50, J=4 | ~86% |
| FedAvg non-IID | Nc=10, J=4 | ~40% |
| FedAvg non-IID | Nc=5, J=4 | ~16% |
| FedAvg non-IID | Nc=1, J=4 | ~7% |
| Task Arith. (Central) | tf=0.2, cr=5 | ~83% |
| Task Arith. (Fed, IID) | tf=0.2, cr=5 | ~86-87% |

### Key Observations

1. **Client drift is severe under heterogeneity**: Performance drops from 84.75% (IID) to below 7% (Nc=1)
2. **Local steps amplify drift**: Increasing J from 4 to 16 causes dramatic accuracy loss
3. **Sensitivity-aware masking helps**: Maintains competitive performance while updating fewer parameters
4. **Magnitude-based selection fails**: Confirms task arithmetic's advantage comes from principled sensitivity estimation

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{penayo2024federated,
  title={Federated Learning Under the Lens of Task Arithmetic},
  author={Penayo, Caio V. M. and Felice, Tiago G. and Turina, Guilherme and EL-KAROUMI, Imane},
  year={2024}
}
```

## References

Key papers that influenced this work:

- [McMahan et al., 2017] Communication-Efficient Learning of Deep Networks from Decentralized Data
- [Ilharco et al., 2023] Editing Models with Task Arithmetic
- [Iurada et al., 2024] Efficient Model Editing with Sensitivity-Aware Sparse Fine-Tuning
- [Karimireddy et al., 2020] SCAFFOLD: Stochastic Controlled Averaging for Federated Learning

## License

This project was developed as part of an academic course. Please contact the authors for usage permissions.

## Contact

For questions or issues, please open an issue on GitHub or contact the project authors.

---

**Note**: This project was developed using Google Colab's free tier. All experiments are designed to run within these computational constraints.
