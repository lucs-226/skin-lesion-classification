import gradio as gr
import torch
from src.config import OUTPUT_DIR, FOLDS, DEVICE, LABELS
from src.model import build_model
from src.data import get_transforms
from src.utils import run_tta

# Load models globally once
models = []
transform = get_transforms('valid')

def load_ensemble():
    if not models:
        for i in range(FOLDS):
            m = build_model()
            # Assumes weights are present (trained or downloaded)
            m.load_state_dict(torch.load(OUTPUT_DIR / f"effnetb3_fold{i}.pth", map_location=DEVICE))
            m.eval()
            models.append(m)

def predict(image):
    if not models: load_ensemble()
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    probs = []
    
    with torch.no_grad():
        for m in models:
            p = run_tta(m, img_tensor)
            probs.append(p.cpu().numpy()[0])
            
    avg = np.mean(probs, axis=0)
    return {LABELS[i]: float(avg[i]) for i in range(len(LABELS))}

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Label())

if __name__ == "__main__":
    load_ensemble()
    demo.launch()
