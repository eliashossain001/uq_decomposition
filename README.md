# UAT-Lite: Uncertainty-Weighted Transformers

**A Lightweight Bayesian Approach for Reliable Language Understanding**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## 📋 Overview

UAT-Lite integrates Bayesian uncertainty quantification directly into transformer architectures through three synergistic components:

1. **Bayesian Embedding Calibration** - Monte Carlo dropout for epistemic uncertainty
2. **Uncertainty-Weighted Attention** - Dynamic token reliability weighting  
3. **Confidence-Guided Decision Shaping** - Risk-aware prediction with abstention

**Key Innovation**: Layer-wise variance decomposition (Theorem 5) enabling interpretable uncertainty attribution across transformer layers.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/uat-lite-main.git
cd repository

# Install dependencies
pip install -r requirements.txt

# Run training on SQuAD 2.0
python scripts/train.py --config configs/experiments/general_nlp.yaml --dataset squad

# Evaluate with uncertainty quantification
python scripts/evaluate.py --checkpoint experiments/general_nlp/squad/checkpoints/best_model.pt
```


## 📁 Project Structure

```
uat-lite-acl/
├── src/                  # Source code
│   ├── models/          # Model implementations
│   ├── data/            # Data loaders
│   ├── evaluation/      # Metrics
│   └── baselines/       # Baseline methods
├── configs/             # Experiment configs
├── scripts/             # Training/evaluation scripts
└── results/             # Output tables & figures
```

## 🔧 Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- Transformers 4.21+
- CUDA 11.2+ (for GPU)

### Setup
```bash
pip install -r requirements.txt
```


## 🧪 Reproducing Results

Run all experiments with a single command:
```bash
bash scripts/reproduce_all.sh
```


## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

