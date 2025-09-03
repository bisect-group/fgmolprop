<div align="center">

# FGMolProp: Functional Group-based Molecular Property Prediction

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

A deep learning framework for molecular property prediction using functional group representations 🧬⚡

</div>

<br>

## 📌 Overview

FGMolProp is a deep learning framework for predicting molecular properties using functional group representations. Unlike traditional molecular representations (SMILES, graphs), this approach decomposes molecules into their constituent functional groups, enabling more interpretable and efficient property prediction.

**Key Features:**
- 🧬 **Functional Group Decomposition**: Novel representation learning using molecular functional groups
- ⚡ **Lightning Fast**: Built on PyTorch Lightning for scalable training
- 🎯 **Multi-Dataset Support**: 35+ molecular property datasets including BBBP, BACE, HIV, QM7/8/9
- 🔧 **Flexible Configuration**: Hydra-based configuration system for easy experimentation
- 📊 **Comprehensive Metrics**: Built-in evaluation metrics for classification and regression tasks
- 🚀 **Pre-training Support**: Self-supervised pre-training for improved performance

**Supported Datasets:**
- **ADMET**: BBBP, BACE, ClinTox, ESOL, FreeSolv, Lipophilicity, SIDER, Tox21, ToxCast
- **Quantum Mechanics**: QM7, QM8, QM9
- **Biochemical**: HIV, MUV, PCBA, ChEMBL
- **Custom**: Easy integration of new datasets

<br>

## 🚀 Installation

### Prerequisites

- Python 3.8+ (3.10 recommended)
- CUDA-capable GPU (optional but recommended)

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/roshanmsb/fgmolprop.git
cd fgmolprop

# Create and activate a virtual environment
python -m venv fgmolprop_env
source fgmolprop_env/bin/activate  # On Windows: fgmolprop_env\Scripts\activate

# Install PyTorch (choose appropriate version for your system)
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt

# Install the project in development mode
pip install -e .
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/roshanmsb/fgmolprop.git
cd fgmolprop

# Create conda environment
conda env create -f environment.yaml -n fgmolprop
conda activate fgmolprop

# Install the project
pip install -e .
```

### Verify Installation

```bash
# Test the installation
python -c "import src; print('Installation successful!')"

# Check available datasets
python src/train.py --help
```

<br>

## 🎯 Training

### Quick Start

Train on BBBP dataset with default configuration:

```bash
# Basic training on CPU
python src/train.py trainer=cpu

# Training on GPU (default)
python src/train.py

# Training with specific dataset
python src/train.py data.dataset=BACE

# Training with custom batch size
python src/train.py data.batch_size=32
```

### Available Experiments

The framework includes pre-configured experiments for all supported datasets:

```bash
# Train on BBBP (Blood-Brain Barrier Penetration)
python src/train.py experiment=bbbp

# Train on BACE (Beta-secretase inhibition)
python src/train.py experiment=bace

# Train on HIV activity prediction
python src/train.py experiment=hiv

# Train on QM9 quantum mechanical properties
python src/train.py experiment=qm9

# Train on Tox21 toxicity prediction
python src/train.py experiment=tox21
```

### Cross-Validation Training

```bash
# 5-fold cross-validation (default)
python src/train.py

# Custom number of folds
python src/train.py n_folds=10

# Train specific fold
python src/train.py data.fold_idx=2
```

### Training with Different Molecular Representations

```bash
# Functional Groups (default)
python src/train.py data.method=FG

# With molecular descriptors
python src/train.py data.descriptors=true

# Different tokenization datasets
python src/train.py data.tokenize_dataset=chembl
```

### Advanced Training Options

```bash
# Multi-GPU training
python src/train.py trainer=ddp trainer.devices=4

# Mixed precision training
python src/train.py trainer=gpu +trainer.precision=16

# Custom learning rate and optimizer
python src/train.py model.optimizer.lr=0.0001

# Enable model compilation (PyTorch 2.0+)
python src/train.py model.compile=true

# Training with weights & biases logging
python src/train.py logger=wandb

# Debug mode (fast training for testing)
python src/train.py debug=default
```

<br>

## 🔬 Pre-training

FGMolProp supports self-supervised pre-training for improved performance:

```bash
# Pre-train on a large unlabeled dataset
python src/pretrain.py data=pretrain

# Pre-train with custom configuration
python src/pretrain.py data=pretrain model=pretrain trainer.max_epochs=100

# Fine-tune from pre-trained checkpoint
python src/train.py experiment=bbbp ckpt_path="path/to/pretrained/model.ckpt"
```

<br>

## ⚙️ Configuration

### Dataset Configuration

Modify `configs/data/default.yaml` or create new dataset configs:

```yaml
_target_: src.data.datamodules.FGRDataModule
data_dir: "${paths.data_dir}"
dataset: "BBBP"           # Dataset name
method: "FG"              # Representation method
descriptors: true         # Include molecular descriptors
tokenize_dataset: "pubchem"  # Tokenization reference
frequency: 500            # Minimum functional group frequency
split_type: "scaffold"    # Data splitting method
batch_size: 16           # Training batch size
num_workers: 4           # Data loading workers
```

### Model Configuration

Modify `configs/model/default.yaml`:

```yaml
_target_: src.models.fgr_module.FGRLitModule

# Optimizer settings
optimizer:
  _target_: src.models.components.losses.SAM  # Sharpness-Aware Minimization
  lr: 0.001
  weight_decay: 0.001
  rho: 0.05

# Model architecture
net:
  _target_: src.models.components.autoencoder.FGRModel
  hidden_dim1: 2048
  hidden_dim2: 1024
  bottleneck_dim: 256
  dropout: 0.1
  activation: "relu"
```

### Supported Datasets

| Dataset | Task Type | Description | Molecules |
|---------|-----------|-------------|-----------|
| BBBP | Classification | Blood-brain barrier penetration | 2,039 |
| BACE | Classification | Beta-secretase inhibition | 1,513 |
| ClinTox | Classification | Clinical toxicity | 1,478 |
| ESOL | Regression | Aqueous solubility | 1,128 |
| FreeSolv | Regression | Hydration free energy | 642 |
| HIV | Classification | HIV replication inhibition | 41,127 |
| Lipophilicity | Regression | Octanol/water distribution | 4,200 |
| SIDER | Classification | Adverse drug reactions | 1,427 |
| Tox21 | Classification | Toxicity on 12 targets | 7,831 |
| QM7 | Regression | Quantum mechanical properties | 7,165 |
| QM8 | Regression | Electronic spectra | 21,786 |
| QM9 | Regression | Quantum properties | 133,885 |

<br>

## 📊 Evaluation

```bash
# Evaluate trained model
python src/eval.py ckpt_path="path/to/checkpoint.ckpt"

# Evaluate on specific dataset
python src/eval.py ckpt_path="path/to/checkpoint.ckpt" data.dataset=BACE

# Generate predictions
python src/eval.py ckpt_path="path/to/checkpoint.ckpt" data.dataset=test_data
```

<br>

## 📈 Monitoring and Logging

### Weights & Biases Integration

```bash
# Configure W&B in configs/logger/wandb.yaml
wandb:
  project: "fgmolprop"
  entity: "your_username"

# Train with W&B logging
python src/train.py logger=wandb
```

### TensorBoard Logging

```bash
# Train with TensorBoard
python src/train.py logger=tensorboard

# View logs
tensorboard --logdir logs/
```

### CSV Logging

```bash
# Simple CSV logging
python src/train.py logger=csv
```

<br>

## 🔍 Hyperparameter Optimization

```bash
# Optuna-based hyperparameter search
python src/train.py -m hparams_search=optuna experiment=bbbp

# Manual parameter sweeps
python src/train.py -m model.optimizer.lr=0.001,0.0001,0.00001 data.batch_size=16,32,64

# Grid search over multiple parameters
python src/train.py -m model.net.hidden_dim1=1024,2048 model.net.dropout=0.1,0.2,0.3
```

<br>

## 📁 Project Structure

```
fgmolprop/
├── configs/                    # Hydra configuration files
│   ├── callbacks/             # Training callbacks
│   ├── data/                  # Dataset configurations
│   ├── experiment/            # Pre-defined experiments
│   ├── logger/                # Logging configurations
│   ├── model/                 # Model configurations
│   ├── trainer/               # Training configurations
│   └── train.yaml             # Main training config
├── data/                      # Data directory
├── logs/                      # Training logs and checkpoints
├── notebooks/                 # Jupyter notebooks for analysis
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model architectures and components
│   ├── utils/                 # Utility functions
│   ├── train.py               # Training script
│   ├── eval.py                # Evaluation script
│   └── pretrain.py            # Pre-training script
├── tests/                     # Test files
├── requirements.txt           # Python dependencies
├── environment.yaml           # Conda environment
└── README.md                  # This file
```

<br>

## 🛠️ Troubleshooting

### Common Issues

**CUDA out of memory:**
```bash
# Reduce batch size
python src/train.py data.batch_size=8

# Enable gradient checkpointing
python src/train.py model.gradient_checkpointing=true
```

**Dataset not found:**
```bash
# Check data directory
python src/train.py paths.data_dir="/path/to/your/data"

# Download datasets automatically (if supported)
python src/train.py data.download=true
```

**Import errors:**
```bash
# Reinstall in development mode
pip install -e .

# Check environment
python -c "import src; print('OK')"
```

**Slow training:**
```bash
# Increase number of workers
python src/train.py data.num_workers=8

# Enable model compilation
python src/train.py model.compile=true

# Use mixed precision
python src/train.py trainer=gpu +trainer.precision=16
```

### Performance Tips

1. **Use appropriate batch size**: Start with 16-32, adjust based on GPU memory
2. **Enable mixed precision**: Add `+trainer.precision=16` for faster training
3. **Use multiple workers**: Set `data.num_workers=4` or higher
4. **Model compilation**: Enable `model.compile=true` with PyTorch 2.0+
5. **GPU selection**: Use `trainer.devices=[0]` to specify GPU

<br>

## 📚 Citation

If you use FGMolProp in your research, please cite:

```bibtex
@software{fgmolprop,
  title={FGMolProp: Functional Group-based Molecular Property Prediction},
  author={Your Name},
  url={https://github.com/roshanmsb/fgmolprop},
  year={2024}
}
```

<br>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<br>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<br>

## 🙏 Acknowledgments

- Built with [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) and [Hydra](https://hydra.cc/)
- Molecular data processing with [RDKit](https://www.rdkit.org/)
- Inspired by advances in molecular representation learning and functional group analysis

<br>

---

**Happy molecular property prediction!** 🧬⚡