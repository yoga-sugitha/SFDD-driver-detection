# SFDD-driver-detection
# SFDD Driver Detection Project

A well-structured deep learning project for State Farm Distracted Driver Detection using PyTorch Lightning.

## Project Structure

```
sfdd_driver_detection/
├── configs/                    # Configuration files
│   ├── base_config.py         # Base configuration classes
│   └── experiment_configs.py  # Experiment-specific configs
├── data/                       # Data handling
│   ├── dataset.py             # Custom dataset classes
│   └── datamodule.py          # Lightning DataModule
├── models/                     # Model architectures
│   ├── resnet.py              # ResNet from scratch
│   ├── pretrained.py          # Pretrained models
│   └── factory.py             # Model factory pattern
├── modules/                    # Training modules
│   └── lightning_module.py    # Lightning training module
├── utils/                      # Utility functions
│   ├── metrics.py             # Performance metrics
│   ├── visualization.py       # Plotting utilities
│   └── gradcam.py             # GradCAM implementation
├── train.py                    # Main training script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Multiple Architectures**: Support for ResNet from scratch and pretrained models
- **Lightning Framework**: Built on PyTorch Lightning for scalability
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and more
- **Explainability**: Integrated GradCAM for model interpretation
- **Performance Analysis**: FLOPs counting and inference latency measurement
- **Flexible Logging**: WandB and CSV logger support
- **Easy Configuration**: Dataclass-based configuration system

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sfdd_driver_detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

Run a single experiment:

```bash
python train.py --exp_id exp001_resnet18_pretrained --max_epochs 30
```

### Run All Experiments

```bash
python train.py --run_all --max_epochs 30
```

### With WandB Logging

```bash
python train.py --exp_id exp001_resnet18_pretrained --enable_wandb
```

### Custom Data Path

```bash
python train.py \
  --exp_id exp001_resnet18_pretrained \
  --data_dir /path/to/your/data \
  --max_epochs 30
```

## Available Experiments

1. **exp001_resnet18_pretrained**: ResNet18 pretrained on ImageNet
2. **exp002_resnet_scratch**: ResNet trained from scratch (small)
3. **exp003_resnet_scratch_deep**: Deeper ResNet from scratch
4. **exp004_sgd_momentum**: ResNet18 with SGD optimizer

## Configuration

Configurations are defined in `configs/experiment_configs.py`. You can:

- Modify existing experiments
- Add new experiment configurations
- Customize data augmentation
- Adjust training hyperparameters

Example configuration:

```python
Config(
    model=ModelConfig(
        model_name="resnet18_pretrained",
        num_classes=10
    ),
    optimizer=OptimizerConfig(
        name="Adam",
        lr=1e-3,
        weight_decay=1e-4
    ),
    training=TrainingConfig(
        max_epochs=30,
        early_stopping_patience=15
    ),
    xai=XAIConfig(enable_xai=True)
)
```

## For Kaggle Notebooks

### Convert to Notebook Cells

If you need to run in Kaggle notebook, you can create cells like:

```python
# Cell 1: Install dependencies
!pip install --quiet lightning wandb ptflops

# Cell 2: Import and setup
from train import run_experiment
from configs.experiment_configs import get_experiment_config

# Cell 3: Run experiment
config = get_experiment_config("exp001_resnet18_pretrained")
config.data.data_dir = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
config.data.num_workers = 0  # Important for Kaggle
config.logging.enable_wandb = False  # Use CSV logger

run_experiment(config, "exp001_resnet18_pretrained")
```

### Memory Management Tips for Kaggle

1. **Set num_workers=0**: Avoid multiprocessing issues
2. **Reduce batch_size**: If running out of memory
3. **Disable GradCAM**: Set `enable_xai=False` for faster execution
4. **Use CSV logger**: Avoid WandB overhead if not needed

## Model Performance Metrics

The framework automatically computes:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Model Complexity**: Parameter count, FLOPs
- **Inference Speed**: Average and std latency per sample
- **Confusion Matrix**: Visual representation of predictions
- **GradCAM**: Visualization of model attention

## Output Structure

```
checkpoints/
└── <experiment_name>/
    ├── best-checkpoint.ckpt    # Best model weights
    └── confusion_matrix.png    # Confusion matrix plot

logs/
└── <experiment_name>/
    └── version_0/
        └── metrics.csv          # Training logs
```

## Extending the Project

### Add a New Model

1. Create model in `models/` directory
2. Register in `models/factory.py`
3. Add configuration in `configs/experiment_configs.py`

### Add Custom Metrics

1. Define metric in `utils/metrics.py`
2. Add to Lightning module in `modules/lightning_module.py`

### Custom Data Augmentation

Modify transforms in `data/datamodule.py`:

```python
self.train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    # Add your transforms here
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Set `num_workers=0`
- Disable GradCAM

### Slow Training
- Increase `num_workers` (if not on Kaggle)
- Use GPU acceleration
- Reduce number of augmentations

### WandB Issues
- Use `wandb_offline=True` in config
- Or switch to CSV logger with `enable_wandb=False`

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@misc{sfdd_driver_detection,
  author = Yoga Sugitha,
  title = {Pipeline for SFDD Driver Detection with PyTorch Lightning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/sfdd_driver_detection}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- State Farm for the dataset
- PyTorch Lightning team
- Weights & Biases for logging tools