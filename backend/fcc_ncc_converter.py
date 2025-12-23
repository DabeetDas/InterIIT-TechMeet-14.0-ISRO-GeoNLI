import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION -------------------------------------------------------

# Replace this with any direct link to a False Color Composite (NIR-Red-Green) image
# This example is a Landsat False Color image (Vegetation appears Red)
IMAGE_URL = "https://i.ibb.co/GfPyGs76/images.jpg"

# Default blue-band coefficients (Patra et al. / IKONOS 11-bit base)
A_B_11BIT = 111.243
B_B = 0.873393
C_B = -0.138254
D_B = -0.100616

# --- 2. CORE LOGIC (Patra et al. Implementation) ----------------------------

def adjust_aB_for_bitdepth(aB_11bit: float, target_bits: int) -> float:
    """Eqn (6): Scale the constant A based on input bit depth."""
    Db = 11 - target_bits
    return aB_11bit * (2.0 ** (-Db))

def infer_bitdepth_from_dtype(arr: np.ndarray) -> int:
    if np.issubdtype(arr.dtype, np.integer):
        return arr.dtype.itemsize * 8
    return 8  # Assume 8-bit for floats/unknowns unless specified

def to_float_radiometric(img: np.ndarray, in_bits: int = None) -> np.ndarray:
    """Normalize input to float representation of raw values."""
    if in_bits is None:
        in_bits = infer_bitdepth_from_dtype(img)
    
    if np.issubdtype(img.dtype, np.floating):
        # If already float, assume it's 0..1, scale up to bit integer range
        return img.astype(np.float32) * (2**in_bits - 1)
    else:
        return img.astype(np.float32)

def fcc_to_ncc(img: np.ndarray, in_bit_depth: int = None) -> np.ndarray:
    """
    Performs the spectral transformation to generate a Natural Color Composite.
    
    Assumptions for Input (FCC):
      Channel 0 (Red displayed)   = Sensor NIR
      Channel 1 (Green displayed) = Sensor Red
      Channel 2 (Blue displayed)  = Sensor Green
      
    Output (NCC):
      Channel 0 = Sensor Red
      Channel 1 = Sensor Green
      Channel 2 = Simulated Blue (Calculated)
    """
    if in_bit_depth is None:
        in_bit_depth = infer_bitdepth_from_dtype(img)

    img_f = to_float_radiometric(img, in_bit_depth)
    
    # Extract Sensor Bands from FCC
    # FCC displays NIR as Red, Red as Green, Green as Blue
    nir_sensor   = img_f[..., 0] 
    red_sensor   = img_f[..., 1]
    green_sensor = img_f[..., 2]

    # Adjust A coefficient for the specific image bit depth
    A_adj = adjust_aB_for_bitdepth(A_B_11BIT, in_bit_depth)

    # Calculate Simulated Blue Band
    # Blue = A + B*Green + C*Red + D*NIR
    sim_blue = A_adj + (B_B * green_sensor) + (C_B * red_sensor) + (D_B * nir_sensor)

    # Clamp values to valid range [0, 2^bits - 1]
    max_val = (2**in_bit_depth - 1)
    sim_blue = np.clip(sim_blue, 0, max_val)

    # Stack to create NCC: (Red_sensor, Green_sensor, Blue_simulated)
    ncc = np.stack([red_sensor, green_sensor, sim_blue], axis=-1)

    # Normalize back to 0..1 for display/saving if desired, or keep as radiometric
    # Here we return 0..255 uint8 for easy display
    ncc_norm = (ncc / max_val * 255).astype(np.uint8)
    
    return ncc_norm

# --- 3. DETECTION UTILITIES (From your snippet) -----------------------------

def _vegetation_mean_ratio_B_over_G(img_f: np.ndarray) -> float:
    B = img_f[..., 2]; G = img_f[..., 1]
    return (B.mean() + 1e-6) / (G.mean() + 1e-6)

def _fraction_B_dominant(img_f: np.ndarray, margin: float = 0.05) -> float:
    R = img_f[..., 0]; G = img_f[..., 1]; B = img_f[..., 2]
    mx = img_f.max() if img_f.max() > 0 else 1.0
    Rn, Gn, Bn = R/mx, G/mx, B/mx
    mask = (Bn > Gn * (1.0 + margin)) & (Bn > Rn * (1.0 + margin))
    return float(mask.mean())

def _ndvi_proxy_fraction(img_f: np.ndarray, threshold: float = 0.15) -> float:
    R = img_f[..., 0]; G = img_f[..., 1] # R=NIR, G=Red in FCC
    denom = (R + G)
    ndvi = np.zeros_like(R, dtype=np.float32)
    nonzero = denom > 1e-6
    ndvi[nonzero] = (R[nonzero] - G[nonzero]) / denom[nonzero]
    return float((ndvi > threshold).mean())

def is_confident_fcc(img, in_bit_depth=None, mean_ratio_threshold=1.0, 
                     frac_B_dom_threshold=0.10, ndvi_frac_threshold=0.10, debug=False):
    # Note: Thresholds lowered slightly for generic web images which might be compressed
    img_f = to_float_radiometric(img, in_bit_depth)
    
    mean_ratio = _vegetation_mean_ratio_B_over_G(img_f)
    # FCCs usually have dominant Red channel (NIR), but the variable names in detection
    # usually assume B/G ratios. 
    # Let's trust the logic: "Vegetation appears Red" means Ch0 is high.
    # The provided detectors check B/G, which might be specific to specific sensor mappings.
    # We will use the NDVI proxy as the strongest indicator.
    
    ndvi_frac = _ndvi_proxy_fraction(img_f, threshold=0.15)
    
    diagnostics = {
        "mean_ratio": mean_ratio,
        "ndvi_frac": ndvi_frac
    }

    # Simplified check for web images: High NDVI proxy usually confirms FCC
    is_fcc = ndvi_frac >= ndvi_frac_threshold
    
    return is_fcc, diagnostics

def safe_fcc_to_ncc(img, in_bit_depth=None, debug=True):
    confident, diag = is_confident_fcc(img, in_bit_depth, debug=debug)
    
    if not confident:
        print("Warning: Image does not appear to be a standard Vegetation-Red FCC.")
        print(f"Diagnostics: {diag}")
        return img, False, diag

    try:
        out = fcc_to_ncc(img, in_bit_depth=in_bit_depth)
        return out, True, diag
    except Exception as e:
        return img, False, {"error": str(e)}

# --- 4. MAIN EXECUTION ------------------------------------------------------

if __name__ == "__main__":
    print(f"Downloading image from: {IMAGE_URL}...")
    
    try:
        # Read image from URL
        original_img = iio.imread(IMAGE_URL)
        
        # Remove alpha channel if present
        if original_img.shape[-1] == 4:
            original_img = original_img[..., :3]

        print("Processing...")
        
        # Attempt conversion
        final_img, was_converted, diagnostics = safe_fcc_to_ncc(original_img)
        
        if was_converted:
            print("Success! Image converted to Natural Color.")
        else:
            print("Skipped conversion (or failed). Showing original.")

        # Display Result
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original (Likely FCC)")
        axes[0].axis('off')
        
        axes[1].imshow(final_img)
        axes[1].set_title("Converted (Simulated NCC)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")