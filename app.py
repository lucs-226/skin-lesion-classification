import gradio as gr
from src.ensemble import EnsemblePredictor
from src.utils import download_weights
from src.config import OUTPUT_DIR, WEIGHTS_ZIP_ID

# 1. Setup: Ensure weights are available
download_weights(WEIGHTS_ZIP_ID, OUTPUT_DIR)

# 2. Initialize Predictor
predictor = EnsemblePredictor()

# 3. Define Interface
demo = gr.Interface(
    fn=predictor.predict_image,
    inputs=gr.Image(type="pil", label="Dermoscopic Image"),
    outputs=gr.Label(num_top_classes=3),
    title="Skin Lesion Ensemble Classifier",
    description="Inference based on EfficientNet-B3 Ensemble (5 Folds) + TTA."
)

if __name__ == "__main__":
    demo.launch()
