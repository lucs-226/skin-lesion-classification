import gradio as gr
import torch
import numpy as np
from PIL import Image
from src.config import OUTPUT_DIR, FOLDS, DEVICE, LABELS, WEIGHTS_ZIP_ID
from src.model import build_model
from src.data import get_transforms
from src.utils import run_tta, download_weights
from src.gradcam import GradCAM, overlay_heatmap

# Global Resources
models = []
gradcam_instance = None
transform = get_transforms('valid')

def load_resources():
    global gradcam_instance
    
    # Ensure weights exist
    download_weights(WEIGHTS_ZIP_ID, OUTPUT_DIR)
    
    if not models:
        for i in range(FOLDS):
            m = build_model()
            m.load_state_dict(torch.load(OUTPUT_DIR / f"effnetb3_fold{i}.pth", map_location=DEVICE))
            m.eval()
            models.append(m)
            
        # Initialize GradCAM on the first model (Fold 0)
        # EfficientNet-B3 last conv layer is usually in .features[-1]
        target_layer = models[0].features[-1]
        gradcam_instance = GradCAM(models[0], target_layer)

def analyze_image(image):
    if not models: load_resources()
    
    # 1. Preprocessing
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad = True # Necessary for GradCAM backward
    
    # 2. Ensemble Prediction (Probabilities)
    probs = []
    with torch.no_grad():
        for m in models:
            p = run_tta(m, img_tensor)
            probs.append(p.cpu().numpy()[0])
    avg_probs = np.mean(probs, axis=0)
    pred_label_idx = np.argmax(avg_probs)
    
    # 3. Grad-CAM Visualization (using Fold 0)
    # Re-enable gradients just for this step on Model 0
    heatmap = gradcam_instance(img_tensor, class_idx=pred_label_idx)
    
    # Resize image to match tensor input size for clean overlay if needed, 
    # or just overlay on original resized to 300x300
    viz_img = image.resize((300, 300))
    overlay = overlay_heatmap(viz_img, heatmap)
    
    return {LABELS[i]: float(avg_probs[i]) for i in range(len(LABELS))}, overlay

# --- Interface ---
demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil", label="Dermoscopic Image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Ensemble Prediction"),
        gr.Image(label="Grad-CAM Explainability")
    ],
    title="Skin Lesion Classification & XAI",
    description="Ensemble Inference with Grad-CAM visualization on the predicted class."
)

if __name__ == "__main__":
    demo.launch()
