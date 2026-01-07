# Multimodal Learning

This repository contains code and notebooks for multimodal learning experiments, including various fusion strategies, ablation studies, and performance assessments.

## Project Structure

```
├── notebooks/                          # Jupyter notebooks for experiments
│   ├── 01_dataset_exploration.ipynb    # Dataset analysis and visualization
│   ├── 02_fusion_comparison.ipynb      # Comparison of fusion strategies
│   ├── 03_strided_conv_ablation.ipynb  # Ablation study on strided convolutions
│   └── 04_final_assessment.ipynb       # Final model evaluation
│
├── src/                                # Source code modules
│   ├── __init__.py                     # Package initialization
│   ├── models.py                       # Neural network architectures
│   ├── datasets.py                     # Dataset utilities and loaders
│   ├── training.py                     # Training and evaluation utilities
│   └── visualization.py                # Visualization functions
│
├── checkpoints/                        # Model checkpoints (created during training)
├── results/                            # Output figures and tables
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. Clone the repository:
```bash
git clone https://github.com/HenriZeiler/Multimodal_Learning.git
cd Multimodal_Learning
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Notebooks

Start Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
# or
jupyter lab
```

Navigate to the `notebooks/` directory and open the desired notebook. The notebooks are numbered in the recommended order of execution:

1. **01_dataset_exploration.ipynb**: Explore and visualize the dataset
2. **02_fusion_comparison.ipynb**: Compare different fusion strategies
3. **03_strided_conv_ablation.ipynb**: Perform ablation studies
4. **04_final_assessment.ipynb**: Final model evaluation and results

### Using Source Modules

The `src/` directory contains reusable Python modules that can be imported in notebooks or scripts:

```python
from src.models import EarlyFusionModel, LateFusionModel
from src.datasets import MultimodalDataset, create_dataloader
from src.training import Trainer, evaluate_model
from src.visualization import plot_training_history, plot_confusion_matrix
```

### Training a Model

Example training script:

```python
from src.models import EarlyFusionModel
from src.datasets import MultimodalDataset, create_dataloader
from src.training import Trainer

# Create dataset and dataloaders
dataset = MultimodalDataset(data, labels)
train_loader = create_dataloader(dataset, batch_size=32)

# Initialize model and trainer
model = EarlyFusionModel(input_dim=512, hidden_dim=256, num_classes=10)
trainer = Trainer(model, device='cuda', learning_rate=0.001)

# Train the model
history = trainer.train(train_loader, val_loader, num_epochs=50)
```

## Project Workflow

1. **Data Exploration**: Use `01_dataset_exploration.ipynb` to understand your data
2. **Model Development**: Experiment with different fusion strategies in `02_fusion_comparison.ipynb`
3. **Ablation Studies**: Analyze model components in `03_strided_conv_ablation.ipynb`
4. **Final Evaluation**: Assess final model performance in `04_final_assessment.ipynb`
5. **Results**: Save figures and tables to the `results/` directory
6. **Checkpoints**: Model checkpoints are automatically saved to `checkpoints/`

## Features

- **Multiple Fusion Strategies**: Early fusion, late fusion, and hybrid approaches
- **Flexible Dataset Handling**: Support for various multimodal data types
- **Training Utilities**: Complete training pipeline with validation and checkpointing
- **Visualization Tools**: Comprehensive plotting functions for analysis
- **Modular Design**: Easy to extend and customize for specific use cases

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the repository owner.