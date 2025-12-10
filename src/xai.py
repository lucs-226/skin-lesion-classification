import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from src import config

class GradCAM:
    """Calculates Gradient-weighted Class Activation Mapping (Heatmaps)."""
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hooks to capture gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        # Forward pass
        output = self.model(x)
        self.model.zero_grad()
        
        # Backward pass on the specific target class
        target = output[0, class_idx]
        target.backward()

        # Generate Heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU to keep only positive contributions
        
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap) # Normalize to 0-1
            
        return heatmap

def plot_xai_analysis(model, img_tensor, true_label, class_map, device):
    """
    Generates a 3-panel plot: Original Image, GradCAM Overlay, Confidence Bar Chart.
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    
    # 1. Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    
    pred_idx = np.argmax(probs)
    
    # 2. GradCAM Calculation (on the last feature layer)
    # Note: For EfficientNet-B3 this is typically model.features[-1]
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # Explain the TRUE class if known, otherwise the PREDICTED class
    target_class = true_label if true_label is not None else pred_idx
    heatmap = grad_cam(img_tensor, target_class)
    
    # 3. Denormalize Image for Display
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    rgb_img = inv_normalize(img_tensor[0]).permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip(rgb_img, 0, 1)
    
    # 4. Overlay Heatmap
    heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    overlay = heatmap_colored * 0.4 + rgb_img
    overlay = overlay / np.max(overlay)
    
    # 5. Plotting
    idx2class = {v: k for k, v in class_map.items()}
    class_names = [idx2class[i] for i in range(len(class_map))]
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # A. Original Image
    ax[0].imshow(rgb_img)
    ax[0].set_title(f"Input Image\nTrue: {idx2class.get(true_label, 'Unknown')}")
    ax[0].axis('off')
    
    # B. GradCAM Focus
    ax[1].imshow(overlay)
    ax[1].set_title(f"GradCAM Focus\nTarget: {idx2class[target_class]}")
    ax[1].axis('off')
    
    # C. Confidence Bars
    y_pos = np.arange(len(class_names))
    sorted_idx = np.argsort(probs)
    
    # Colors: Green if correct/truth, Red if wrong prediction
    colors = ['gray'] * len(class_names)
    if true_label is not None:
        colors[true_label] = 'green'
    colors[pred_idx] = 'red' if pred_idx != true_label else 'green'

    ax[2].barh(y_pos, probs[sorted_idx], color=[colors[i] for i in sorted_idx])
    ax[2].set_yticks(y_pos)
    ax[2].set_yticklabels([class_names[i] for i in sorted_idx])
    ax[2].set_title("Model Confidence")
    ax[2].set_xlabel("Probability")
    
    plt.tight_layout()
    plt.show()
