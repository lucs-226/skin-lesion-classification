import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

def analyze_class_imbalance(df, class_col='dx'):
    plt.figure(figsize=(10, 8))
    counts = df[class_col].value_counts().sort_values(ascending=False)
    total = len(df)
    
    explode = [0.05] * len(counts)
    explode[0] = 0.15
    # Use a colormap
    colors = plt.cm.Spectral(np.linspace(0, 1, len(counts)))
    
    plt.pie(
        counts.values,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        explode=explode,
        shadow=True,
        colors=colors,
        labels=counts.index
    )
    plt.title(f"Class Distribution (Total: {total})")
    plt.show()

def show_class_samples(df, class_col='dx', path_col='path'):
    classes = sorted(df[class_col].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(20, 3))
    
    for i, cls in enumerate(classes):
        subset = df[df[class_col] == cls]
        if len(subset) > 0:
            sample_path = subset[path_col].iloc[0]
            img = Image.open(sample_path)
            axes[i].imshow(img)
            axes[i].set_title(cls)
            axes[i].axis('off')
    plt.show()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
