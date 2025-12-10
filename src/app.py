import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from src import config, model, data

# Singleton Model Instance
MODEL = None
# Static labels for HAM10000
LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def get_predictor():
    """Loads the model into memory only once (Singleton Pattern)."""
    global MODEL
    if MODEL is None:
        # Initialize Architecture
        MODEL = model.build_model(config.NUM_CLASSES, pretrained=False)
        # Load weights (CPU is sufficient for single-image inference demo)
        checkpoint = config.OUTPUT_PATH / "effnetb3_fold0.pth"
        
        if checkpoint.exists():
            print(f"Loading weights from {checkpoint}...")
            MODEL.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        else:
            print("WARNING: Checkpoint not found. Using random weights for demo.")
            
        MODEL.eval()
    return MODEL

def predict_image(image):
    """Inference function for Gradio."""
    if image is None: return None
    image = image.convert('RGB')
    
    # 1. Preprocessing (Same as validation pipeline)
    _, val_tf = data.get_transforms(config.IMG_SIZE)
    img_tensor = val_tf(image).unsqueeze(0)
    
    # 2. Inference
    net = get_predictor()
    with torch.no_grad():
        outputs = net(img_tensor)
        probs = F.softmax(outputs, dim=1).squeeze().numpy()
    
    # 3. Format output for gr.Label
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# Gradio Interface Definition
if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Dermoscopic Image"),
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="Skin Cancer Classification Demo",
        description="Upload a dermoscopic image to classify the lesion type using EfficientNet-B3.",
        allow_flagging="never"
    )
    
    print("Launching Gradio Demo...")
    demo.launch(share=True)
