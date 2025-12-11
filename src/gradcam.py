import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # 2. Backward Pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # 3. Generate Heatmap
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global Average Pooling of Gradients (Weights)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of feature maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU & Normalization
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

def overlay_heatmap(img_pil, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlays the heatmap on the original PIL image."""
    img_np = np.array(img_pil)
    
    # Convert heatmap to RGB 
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay
