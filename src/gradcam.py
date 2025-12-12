import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        # Forward
        output = self.model(x)
        self.model.zero_grad()
        
        # Backward target
        target = output[0, class_idx]
        target.backward()

        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

def overlay_heatmap(img_pil, heatmap):
    img_np = np.array(img_pil)
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Convert to RGB color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return overlay
