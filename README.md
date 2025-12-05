# SFDD-driver-detection
# SFDD Driver Detection Project

A well-structured deep learning project for State Farm Distracted Driver Detection using PyTorch Lightning and Hydra.


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
python main.py 
```

### Run Multiple Experiments

```bash
python main.py -m \
    data.task_type=binary,multiclass \
    model=resnet_scratch,resnet \
    training=epoch30
```


## Available Experiments

1. **exp001_resnet18_pretrained**: ResNet18 pretrained on ImageNet
2. **exp002_resnet_scratch**: ResNet trained from scratch (small)

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
3. Add configuration in `configs`

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
  title = Pipeline for SFDD Driver Detection with PyTorch Lightning and Hydra,
  year = 2025,
  publisher = GitHub,
  url = {https://github.com/yoga-sugitha/sfdd_driver_detection}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- State Farm for the dataset
- PyTorch Lightning team
- Weights & Biases for logging tools
- Hydra for complex experiment management