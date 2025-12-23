import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"  # Path to your trained model
# Replace this with your actual image URL
IMAGE_URL = "https://files.catbox.moe/r5gjug.jpg" 
CONFIDENCE_THRESHOLD = 0.880237  # Adjust this based on your validation results

def classify_url(url: str, model_path: str, threshold: float = 0.880237):
    """
    Downloads an image from a URL and classifies it as FCC or NCC.
    """
    try:
        # 1. Download the image
        print(f"Downloading image from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("Image downloaded successfully")
        
        # 2. Load Model
        model = YOLO(model_path)
        
        # 3. Determine Class Indices (Dynamic)
        names = model.names
        fcc_idx, ncc_idx = None, None
        
        for idx, name in names.items():
            if "FCC" in str(name).upper(): fcc_idx = idx
            elif "NCC" in str(name).upper(): ncc_idx = idx
            
        # Fallback if names aren't explicit
        if fcc_idx is None: fcc_idx = 1
        if ncc_idx is None: ncc_idx = 0
            
        # 4. Inference
        results = model(image, verbose=False)
        probs = results[0].probs.data
        
        fcc_prob = probs[fcc_idx].item()
        ncc_prob = probs[ncc_idx].item() if ncc_idx is not None else (1.0 - fcc_prob)

        # 5. Classification Logic
        if fcc_prob >= threshold:
            prediction = "FCC"
            confidence = fcc_prob
        else:
            prediction = "NCC"
            confidence = ncc_prob

        # 6. Output
        print("-" * 30)
        print(f"Prediction:  {prediction}")
        print(f"Confidence:  {confidence:.4f}")
        print(f"FCC Prob:    {fcc_prob:.4f}")
        print(f"Threshold:   {threshold}")
        print("-" * 30)

        return prediction, confidence

    except Exception as e:
        print(f"Error: {e}")
        return None, 0.0

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    classify_url(IMAGE_URL, MODEL_PATH, CONFIDENCE_THRESHOLD)