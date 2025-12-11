import torch
import numpy as np
from src.config import DEVICE, FOLDS, OUTPUT_DIR, LABELS
from src.model import build_model
from src.dataset import get_transforms

class EnsemblePredictor:
    def __init__(self, weights_dir=OUTPUT_DIR):
        self.device = DEVICE
        self.weights_dir = weights_dir
        
        # Initialize architecture once
        self.model = build_model(num_classes=len(LABELS), pretrained=False)
        self.transform = get_transforms(mode='valid')
        
    def predict_image(self, image_pil):
        """
        Performs inference using 5-Fold Ensemble + TTA (Test Time Augmentation).
        Returns a dictionary of class probabilities.
        """
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        ensemble_probs = np.zeros(len(LABELS))
        valid_folds = 0
        
        # Iterate over all 5 folds (Weight Swapping)
        for i in range(FOLDS):
            w_path = self.weights_dir / f"effnetb3_fold{i}.pth"
            
            if not w_path.exists():
                print(f"Warning: Weight file {w_path} not found. Skipping fold.")
                continue
                
            # Load weights into the model
            state_dict = torch.load(w_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            with torch.no_grad():
                # TTA: Average of Original + Horizontal Flip
                p1 = self.model(img_tensor).softmax(1)
                p2 = self.model(torch.flip(img_tensor, dims=[3])).softmax(1)
                avg_p = (p1 + p2) / 2.0
                
            ensemble_probs += avg_p.cpu().numpy()[0]
            valid_folds += 1
            
        if valid_folds == 0:
            raise FileNotFoundError(f"No weight files found in {self.weights_dir}")
            
        # Final averaging
        final_probs = ensemble_probs / valid_folds
        return {LABELS[i]: float(final_probs[i]) for i in range(len(LABELS))}
