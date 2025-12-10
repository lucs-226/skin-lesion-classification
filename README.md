# Skin Lesion Classification (EfficientNet-B3)

An end-to-end Deep Learning pipeline for multi-class skin cancer classification using the **HAM10000** dataset. This repository implements a robust training strategy focusing on reproducibility, data leakage prevention, and class imbalance handling.

### Project Objective
The primary goal is to classify dermoscopic images into one of 7 diagnostic categories (e.g., Melanoma, Nevus, Basal Cell Carcinoma).
Beyond high accuracy, the engineering objective was to build a **modular, production-ready codebase** capable of iterating quickly over different architectures and hyperparameters.

### Model Architecture: Why EfficientNet?
We selected **EfficientNet-B3** as the backbone for this task.

* **Compound Scaling:** Unlike ResNets, which scale depth/width arbitrarily, EfficientNet uniformly scales depth, width, and resolution. This results in better feature extraction for the specific input size ($300 \times 300$).
* **Parameter Efficiency:** B3 offers a superior accuracy-to-parameter ratio compared to VGG or ResNet50, reducing training time and inference latency without sacrificing performance.
* **Transfer Learning:** Pre-trained weights (ImageNet) allow the model to leverage low-level feature detectors (edges, textures) immediately, which is crucial given the limited size of medical datasets.

### Strategy & Methodology

### 1. Data Integrity & Leakage Prevention
The HAM10000 dataset contains multiple images of the *same* lesion. A naive random split would place the same lesion in both Train and Validation sets, leading to **Data Leakage** and inflated metrics.
* **Solution:** We deduplicated the dataset based on `lesion_id`, ensuring that all images of a specific lesion reside strictly in one fold.

### 2. Handling Class Imbalance
The dataset is heavily skewed towards Nevi ($nv$) and huge class disparity exists for rare classes like Dermatofibroma ($df$).
* **Weighted Random Sampling:** We implemented a custom sampler that oversamples minority classes during batch generation.
* **Focal Loss:** Replaced standard Cross Entropy with **Focal Loss** ($\gamma=2.0$). This dynamically down-weights easy examples (background classes) and forces the model to focus on hard, misclassified examples.

### 3. Regularization & Augmentation
To prevent overfitting on the training set:
* **TTA (Test Time Augmentation):** (Optional in inference) Averaging predictions over augmented versions of the input.
* **Transforms:** Rotation, flipping, and color jittering to force the model to learn invariant features rather than memorizing orientation or lighting.

---

### The Challenge: Domain Shift & Generalization
While the model achieves high metrics (Accuracy/F1) on the validation set, we observed a **Domain Shift** phenomenon when evaluating on external data or conceptually different subsets.

### The Problem
Medical images are highly sensitive to the acquisition device (camera sensor, lighting, dermoscope type). The model tends to learn "shortcuts"—associating specific lighting conditions or artifacts (e.g., rulers, gel bubbles) with a class, rather than the pathology itself.
* **Observation:** The distribution of the training data (HAM10000) does not perfectly match real-world clinical inputs (Covariate Shift).
* **Impact:** Validation scores are likely optimistic compared to real-world deployment performance.

### Mitigation Strategies Implemented
1.  **Heavy Color Jittering:** Randomizing brightness and contrast during training to reduce dependency on specific lighting conditions.
2.  **Stratified K-Fold:** validating across 5 different splits to ensure the model isn't just lucky on one specific data subset.
3.  **Conservative Model Selection:** We save checkpoints based on Validation Loss (a proxy for calibration) rather than just Accuracy.

---

### Repository Structure
```text
skin-cancer-classification/
├── data/                   # Dataset (HAM10000)
├── output/                 # Model checkpoints & Logs
├── src/
│   ├── config.py           # Centralized Hyperparameters
│   ├── data.py             # Data Loading & Preprocessing
│   ├── model.py            # EfficientNet-B3 Definition
│   ├── train.py            # Training Loop (K-Fold)
│   ├── predict.py          # Inference Script
│   └── utils.py            # Helper functions
