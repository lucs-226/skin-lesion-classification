# 🔬 Skin Lesion Classification Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready pipeline for skin lesion classification using the HAM10000 dataset. This project implements a complete 20-step preprocessing and training pipeline with stratified splitting, data augmentation, class balancing, and support for both PyTorch and TensorFlow/Keras.

## 📋 Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Complete 20-step preprocessing pipeline** with data validation and quality checks
- **Stratified train/val/test split** (70/15/15) with lesion_id integrity
- **Multiple preprocessing options**: basic resize, contrast enhancement, CLAHE
- **Advanced data augmentation**: rotation, flip, brightness adjustment
- **Class balancing** through targeted augmentation and class weights
- **Support for both PyTorch and TensorFlow/Keras**
- **Comprehensive visualizations** and analysis reports
- **Modular, reusable code** with extensive documentation
- **Unit tests** for all major components
- **CLI tools** for easy automation

## 📊 Dataset

This project uses the **HAM10000** (Human Against Machine with 10000 training images) dataset, which contains dermatoscopic images of pigmented skin lesions.

### Classes (7 types):
- `nv` - Melanocytic nevi (66.9%)
- `mel` - Melanoma (11.1%)
- `bkl` - Benign keratosis (11.0%)
- `bcc` - Basal cell carcinoma (5.1%)
- `akiec` - Actinic keratoses (3.3%)
- `vasc` - Vascular lesions (1.4%)
- `df` - Dermatofibroma (1.1%)

**Download:** [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/skin-lesion-classification.git
cd skin-lesion-classification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 🏃 Quick Start

### 1. Download the Dataset

```bash
# Automated download (requires Kaggle API credentials)
bash scripts/download_data.sh

# Or manually download from Kaggle and place in data/ directory
```

### 2. Run the Complete Pipeline

```bash
# Run full preprocessing pipeline
python scripts/run_pipeline.py

# Or use the interactive notebook
jupyter notebook notebooks/02_preprocessing_pipeline.ipynb
```

### 3. Train a Model

```bash
# Train with default configuration
python scripts/train_model.py --config configs/config.yaml

# Train with custom settings
python scripts/train_model.py --model resnet50 --epochs 50 --batch-size 32
```

## 🔄 Pipeline Overview

The preprocessing pipeline consists of 20 comprehensive steps:

### Phase 1: Data Loading & Validation (Steps 1-2)
1. Load images and CSV metadata
2. Merge data, check for duplicates and missing files

### Phase 2: Analysis (Steps 3-5)
3. Analyze class distribution
4. Evaluate metadata (age, sex, localization)
5. Check image quality and dimensions

### Phase 3: Dataset Split (Steps 6-7)
6. Perform stratified split (70/15/15)
7. Maintain lesion_id integrity (prevent data leakage)

### Phase 4: Preprocessing (Steps 8-11)
8. Image loading and resizing (224×224)
9. Normalization (ImageNet or [0,1])
10. Data augmentation (training only)
11. Validation/test: resize + normalize only

### Phase 5: Class Balancing (Steps 12-18)
12. Encode classes to integers/one-hot
13. Analyze class imbalance
14. Apply targeted augmentation for rare classes
15. Oversample minority classes
16. Calculate inverse frequency class weights
17. Optional: balanced batch sampler
18. Document before/after balancing

### Phase 6: Output (Steps 19-20)
19. Save processed datasets and configurations
20. Generate visualizations and reports

## 📁 Project Structure

```
skin-lesion-classification/
├── src/                        # Source code
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Image preprocessing
│   ├── augmentation.py        # Data augmentation
│   ├── dataset.py             # PyTorch Dataset classes
│   ├── models.py              # Model architectures
│   ├── train.py               # Training logic
│   ├── evaluate.py            # Evaluation metrics
│   └── utils.py               # Helper functions
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── scripts/                    # Executable scripts
│   ├── run_pipeline.py        # Run full pipeline
│   ├── train_model.py         # Train models
│   └── download_data.sh       # Download dataset
├── configs/                    # Configuration files
│   ├── config.yaml            # Main config
│   └── model_config.yaml      # Model hyperparameters
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── data/                       # Dataset (gitignored)
├── outputs/                    # Generated outputs
└── models/                     # Saved models

```

## 💻 Usage

### Using as Python Module

```python
from src.data_loader import load_ham10000_data
from src.preprocessing import preprocess_image
from src.augmentation import augment_image
from src.dataset import SkinLesionDataset

# Load data
df = load_ham10000_data('data/HAM10000_metadata.csv', 
                         image_dirs=['data/images_part_1', 'data/images_part_2'])

# Preprocess single image
img = preprocess_image('path/to/image.jpg', 
                       target_size=(224, 224),
                       normalize='imagenet')

# Create PyTorch dataset
from torch.utils.data import DataLoader
dataset = SkinLesionDataset('balanced_train_set.csv', 
                            transform=train_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Using CLI Scripts

```bash
# Run pipeline with custom settings
python scripts/run_pipeline.py \
    --data-path data/ \
    --output-path outputs/ \
    --target-size 224 \
    --normalize imagenet

# Train model
python scripts/train_model.py \
    --model resnet50 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --use-class-weights
```

### Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open any notebook in notebooks/ directory
# Follow the step-by-step instructions
```

## 📈 Results

The pipeline generates comprehensive visualizations and reports:

### Generated Files

**Visualizations:**
- `1_class_distribution.png` - Original class distribution
- `2_metadata_analysis.png` - Age, sex, localization patterns
- `3_sample_images_per_class.png` - Visual examples
- `4_preprocessing_variants.png` - Preprocessing comparisons
- `5_augmentation_examples.png` - Augmentation demonstrations
- `6_class_balancing_comparison.png` - Before/after balancing
- `7_complete_pipeline_summary.png` - Comprehensive overview

**Data Files:**
- `balanced_train_set.csv` - Balanced training set
- `val_set.csv` - Validation set
- `test_set.csv` - Test set
- `class_mappings.json` - Class encodings and weights
- `pipeline_summary_report.txt` - Detailed text report

**Code Templates:**
- `pytorch_dataloader_template.py` - PyTorch implementation
- `keras_datagenerator_template.py` - TensorFlow/Keras implementation

### Example Metrics

| Metric | Before Balancing | After Balancing |
|--------|-----------------|-----------------|
| Imbalance Ratio | 58.3:1 | 1:1 |
| Training Samples | 7,010 | ~46,900 |
| Minority Class (df) | 80 images | 6,705 images |

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [**PIPELINE.md**](docs/PIPELINE.md) - Complete pipeline documentation
- [**API.md**](docs/API.md) - API reference for all modules
- [**RESULTS.md**](docs/RESULTS.md) - Benchmark results and analysis

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src tests/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to:
- Update tests as appropriate
- Update documentation
- Follow PEP 8 style guidelines
- Add descriptive commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HAM10000 Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).
- **Kaggle**: For hosting the dataset
- **PyTorch & TensorFlow**: For deep learning frameworks

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Project Link**: [https://github.com/yourusername/skin-lesion-classification](https://github.com/yourusername/skin-lesion-classification)

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

**Made with ❤️ for medical image analysis**
