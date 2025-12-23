import cv2
import numpy as np

def preprocess_image(img):
    """Preprocesses the image for the model"""
    img = edge_enhancement(img)
    img = apply_clahe(img)
    img = dcp_dehazing(img)
    return img
    
def edge_enhancement(img, alpha=1.5, sigma=1.0):
    """Sharpens object boundaries using unsharp masking"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Create unsharp mask
    unsharp_mask = gray.astype(np.float32) - blurred.astype(np.float32)
    
    # Apply to each channel
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        channel = img[:, :, i].astype(np.float32)
        enhanced = channel + alpha * unsharp_mask
        result[:, :, i] = np.clip(enhanced, 0, 255)
    
    return result.astype(np.uint8)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhances local contrast and detail using CLAHE"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_enhanced = clahe.apply(l_channel)
    
    # Merge channels and convert back
    lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return result

def dcp_dehazing(img, omega=0.95, t0=0.1, patch_size=15):
    """Removes haze using Dark Channel Prior"""
    img_float = img.astype(np.float32)
    
    # Compute dark channel
    dark = dark_channel(img, patch_size)
    
    # Estimate atmospheric light
    A = estimate_atmospheric_light(img, dark)
    
    # Normalize image
    img_norm = img_float / (A + 1e-8)
    
    # Compute transmission map
    dark_norm = dark_channel(img_norm.astype(np.uint8), patch_size)
    transmission = 1 - omega * (dark_norm.astype(np.float32) / 255.0)
    transmission = np.clip(transmission, t0, 1.0)
    
    # Dehaze
    result = np.zeros_like(img_float)
    for i in range(3):
        result[:, :, i] = ((img_float[:, :, i] - A[i]) / (transmission + 1e-8)) + A[i]
    
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

def dark_channel(img, patch_size=15):
    """Compute dark channel of image"""
    b, g, r = cv2.split(img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(dc, kernel)
    return dark

def estimate_atmospheric_light(img, dark, percentile=0.1):
    """Estimate atmospheric light from dark channel"""
    h, w = dark.shape
    num_pixels = int(h * w * percentile)
    dark_flat = dark.ravel()
    indices = dark_flat.argsort()[-num_pixels:]
    
    img_flat = img.reshape(-1, 3)
    atmospheric_light = np.mean(img_flat[indices], axis=0)
    return atmospheric_light
