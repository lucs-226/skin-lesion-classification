import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PLOTS_DIR, LABELS

def plot_class_distribution(df, target_col='label', title='Class Distribution'):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target_col, data=df, order=LABELS)
    plt.title(title)
    plt.savefig(PLOTS_DIR / "class_distribution.png")
    plt.close()

def plot_confusion_matrix(cm, filename="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(PLOTS_DIR / filename)
    plt.close()
