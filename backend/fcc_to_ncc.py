# this file is refernced from main don't delete it.
import io
import uuid
import numpy as np
import imageio.v3 as iio
from PIL import Image
from supabase import create_client, Client

# --- 1. CONFIGURATION -------------------------------------------------------

# SUPABASE CREDENTIALS (REPLACE THESE)
SUPABASE_URL = "https://lrwgneqmozlynjwoelrx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxyd2duZXFtb3pseW5qd29lbHJ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ3MTQ5MzMsImV4cCI6MjA4MDI5MDkzM30.6yytOlsqjljrAqKfO9kdbJCSHYjRpqu93GXL09s3MoE"
BUCKET_NAME = "images" # Ensure this bucket is created and 'public'

# Default blue-band coefficients (Patra et al.)
A_B_11BIT = 111.243
B_B = 0.873393
C_B = -0.138254
D_B = -0.100616

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. MATH HELPERS (Core Logic) -------------------------------------------

def _transform_matrix(img: np.ndarray) -> np.ndarray:
    """Internal helper: Performs the NumPy spectral transformation."""
    
    # 1. Infer Bit Depth & Normalize
    is_int = np.issubdtype(img.dtype, np.integer)
    in_bits = img.dtype.itemsize * 8 if is_int else 8
    
    if np.issubdtype(img.dtype, np.floating):
        img_f = img.astype(np.float32) * (2**in_bits - 1)
    else:
        img_f = img.astype(np.float32)

    # 2. Extract Sensor Bands (FCC: NIR->R, Red->G, Green->B)
    nir_sensor   = img_f[..., 0] 
    red_sensor   = img_f[..., 1]
    green_sensor = img_f[..., 2]

    # 3. Calculate Simulated Blue
    # Eqn 6: Scale A based on bit depth
    Db = 11 - in_bits
    A_adj = A_B_11BIT * (2.0 ** (-Db))
    
    sim_blue = A_adj + (B_B * green_sensor) + (C_B * red_sensor) + (D_B * nir_sensor)
    
    max_val = (2**in_bits - 1)
    sim_blue = np.clip(sim_blue, 0, max_val)

    # 4. Stack NCC (R, G, B)
    ncc = np.stack([red_sensor, green_sensor, sim_blue], axis=-1)

    # 5. Normalize to uint8 (0-255) for standard image formats
    return (ncc / max_val * 255).astype(np.uint8)

# --- 3. MAIN FUNCTION -------------------------------------------------------

def fcc_to_ncc(img_url: str) -> str:
    """
    Downloads an FCC image, converts it to NCC, uploads to Supabase, 
    and returns the public URL.
    """
    try:
        # 1. Download Image
        print(f"Fetching: {img_url}...")
        img_arr = iio.imread(img_url)

        # Handle Alpha Channel
        if img_arr.shape[-1] == 4:
            img_arr = img_arr[..., :3]

        # 2. Perform Transformation
        ncc_arr = _transform_matrix(img_arr)

        # 3. Convert NumPy Array to Image Bytes (JPEG)
        img_pil = Image.fromarray(ncc_arr)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=95)
        file_bytes = buffer.getvalue()

        # 4. Upload to Supabase
        # Generate unique filename to avoid collisions
        file_name = f"ncc_converted_{uuid.uuid4()}.jpg"
        
        print("Uploading to Supabase...")
        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_name, 
            file=file_bytes, 
            file_options={"content-type": "image/jpeg"}
        )

        # 5. Retrieve Public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
        return public_url

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- 4. EXECUTION -----------------------------------------------------------

if __name__ == "__main__":
    # Example FCC Image (Vegetation is Red)
    fcc_url = "https://i.ibb.co/KpW7YmQM/download.png"
    
    result_url = fcc_to_ncc(fcc_url)
    
    if result_url:
        print("\nProcess Complete!")
        print(f"NCC Image URL: {result_url}")
    else:
        print("\nFailed to convert and upload.")