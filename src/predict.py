import torch
import argparse
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from src import config, model, data

ID2LABEL = {
    0: 'akiec', 1: 'bcc', 2: 'bkl', 
    3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'
}

def load_trained_model(checkpoint_path):
    device = config.DEVICE
    net = model.build_model(config.NUM_CLASSES, pretrained=False)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.to(device)
    net.eval()
    return net

def predict(image_path, net):
    _, val_tf = data.get_transforms(config.IMG_SIZE)
    img = Image.open(image_path).convert('RGB')
    img_tensor = val_tf(img).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = net(img_tensor)
        probs = F.softmax(outputs, dim=1)
        
    prob, idx = torch.max(probs, 1)
    return ID2LABEL[idx.item()], prob.item()

def main():
    parser = argparse.ArgumentParser(description="Skin Cancer Inference")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth checkpoint")
    
    args = parser.parse_args()
    
    if not Path(args.img).exists():
        raise FileNotFoundError(f"Image not found: {args.img}")
        
    net = load_trained_model(args.model)
    cls_name, confidence = predict(args.img, net)
    
    print(f"\nPrediction: {cls_name.upper()}")
    print(f"Confidence: {confidence:.4f}\n")

if __name__ == "__main__":
    main()
