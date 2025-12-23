"""
Flask application for VRSBench pipeline API.

This module provides REST API endpoints for processing query JSON files
through the VRSBench pipeline.
"""

import logging
import sys
import requests
import io
from flask import Flask, request, jsonify
from PIL import Image
from SAR_FCC import fcc, sar
from fcc_to_ncc import fcc_to_ncc
from caption import caption
from grounding import calculate_obb_iou, grounding, parse_qwen_response, run_yolo_obb
from binary_vqa import binary_vqa
from modal_request import qwen_base, qwen_ft_ground
from numeric_vqa import numeric_vqa
from semantic_vqa import semantic_vqa
from general_vqa import general_vqa
from overall_classifier import classify_query_from_messages, QueryClassification
from agriculture import get_model_response
import ast
from ultralytics import YOLO
import cv2
import numpy as np
import urllib.request
from classify_fcc_ncc import classify_url
import uuid
from supabase import create_client, Client
from flask_cors import CORS
from dotenv import load_dotenv
from typing import Tuple
import os
import math
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))
BUCKET_NAME = "images" # Ensure this bucket is created and 'public'

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)



# load commmon YOLO models for everything
current_dir = os.path.dirname(os.path.abspath(__file__))
best_model_path = os.path.join(current_dir, "best.pt")
best_model = YOLO(best_model_path)

runs_obb_model_path = os.path.join(current_dir, "runs_obb_train4_weights_last.pt")
runs_obb_model = YOLO(runs_obb_model_path)
last_model_path = os.path.join(current_dir, "last.pt")
last_model = YOLO(last_model_path)
base_yolo_path = os.path.join(current_dir, "yolo11x-obb.pt")
base_yolo = YOLO(base_yolo_path)
yolo_model_1 = runs_obb_model
yolo_model_2 = last_model
yolo_base = base_yolo

def url_to_image(url):
    """
    Downloads an image from a URL and converts it to OpenCV format.
    
    Args:
        url (str): URL of the image
        
    Returns:
        numpy.ndarray: Image in BGR format
    """
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

def get_red_percentage(image_path):
    """
    Returns the percentage of red pixels (vegetation) in an FCC image.
    
    Args:
        image_path (str): Path or URL to the FCC image
        
    Returns:
        float: Percentage of red pixels (0-100), or -1 on error
    """
    img_bgr = url_to_image(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return -1
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = img_rgb.shape
    total_pixels = height * width
    # Extract RGB channels
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]

    # Red pixels: high red, moderate green, low blue
    # Thresholds: red > 50, red > blue
    red_mask = (r_channel > 50) & (b_channel < r_channel)
    
    red_pixel_count = np.sum(red_mask)
    
    red_percentage = (red_pixel_count / total_pixels) * 100
    return red_percentage

def get_blue_percentage(image_path):
    """
    Returns the percentage of blue/cyan pixels (barren land) in an FCC image.
    
    Args:
        image_path (str): Path or URL to the FCC image
        
    Returns:
        float: Percentage of blue/cyan pixels (0-100), or -1 on error
    """
    img_bgr = url_to_image(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return -1
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = img_rgb.shape
    total_pixels = height * width
    
    # Extract RGB channels
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]
    
    # Blue/cyan pixels: high blue, moderate green, low red
    # Thresholds: blue > 80, blue >= red
    blue_mask = (b_channel > 80) & (r_channel < b_channel)
    blue_pixel_count = np.sum(blue_mask)
    
    blue_percentage = (blue_pixel_count / total_pixels) * 100
    return blue_percentage

def image_classify(image_path, model: YOLO, yolo_threshold=0.880237):
    try:
        # 1. Image Acquisition
        img_array = np.asarray(bytearray(requests.get(image_path, timeout=10).content), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None: return "NCC"
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. SAR Detection (Low Saturation)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        if np.mean(img_hsv[:, :, 1]) < 15:
            return "SAR"

        try:
            classification = classify_url(image_path, model, yolo_threshold)
            prediction, confidence = classification
            if prediction == "FCC":
                return "FCC"
            else:
                return "NCC"
        except Exception:
            return "NCC"

    except Exception:
        return "NCC"

def convert_cv2_angle_to_frontend_format(angle_cv2: float) -> float:
    """
    Convert angle from cv2.minAreaRect format ([-90, 0) degrees) to frontend format ([0, 180) degrees).
    
    cv2.minAreaRect returns angles in the range [-90, 0) degrees.
    The frontend expects angles in the range [0, 180) degrees.
    
    Args:
        angle_cv2: Angle in cv2.minAreaRect format ([-90, 0) degrees)
    
    Returns:
        Angle in frontend format ([0, 180) degrees)
    """
    # Convert from [-90, 0) to [0, 180) format
    # Formula: (angle + 90) % 180
    # This converts: -90 → 0, -45 → 45, 0 → 90
    angle_frontend = (angle_cv2 + 90) % 180
    return angle_frontend

def convert_obb_to_point_representation(obbs: list) -> list:
    """
    Convert oriented bounding box from [cx, cy, w, h, theta] format to point-based representation.
    
    Args:
        obb: [cx, cy, w, h, theta] where:
            - cx, cy are normalized center coordinates (0-1)
            - w, h are normalized width and height (0-1)
            - theta is angle in degrees
    
    Returns:
        List of 8 values [x1, y1, x2, y2, x3, y3, x4, y4] in normalized coordinates (0-1),
        ordered clockwise.
    """
    results = []
    for obb_obj in obbs:
        cx_norm, cy_norm, w_norm, h_norm, theta_deg = obb_obj['obbox']
    
        # --- FIX: Offset the angle by 90 degrees ---
        # If your boxes are 90 deg off, it's likely a mismatch between 
        # "0 degrees = Horizontal" (Math) vs "0 degrees = Vertical" (Your Data).
        theta_deg = theta_deg + 90
        
        # Convert angle from degrees to radians
        theta_rad = math.radians(theta_deg)
        
        # Calculate half-width and half-height in normalized coordinates
        dx = w_norm / 2.0
        dy = h_norm / 2.0
        
        # Define the 4 corners relative to center (before rotation)
        # Order: top-left, top-right, bottom-right, bottom-left
        corners_relative = np.array([
            [-dx, -dy],  # top-left
            [dx, -dy],   # top-right
            [dx, dy],    # bottom-right
            [-dx, dy]    # bottom-left
        ], dtype=float)
        
        # Rotation matrix
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        R = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], dtype=float)
        
        # Rotate corners
        corners_rotated = corners_relative @ R.T
        
        # Translate to center position (in normalized coordinates)
        corners_absolute = corners_rotated + np.array([cx_norm, cy_norm])
        
        # Ensure all coordinates are within [0, 1] range
        corners_absolute = np.clip(corners_absolute, 0.0, 1.0)
        
        # Flatten to [x1, y1, x2, y2, x3, y3, x4, y4] format
        result = corners_absolute.flatten().tolist()

        result_obj = {
            "object-id": obb_obj['object-id'],
            "obbox": result
        }
        results.append(result_obj)
    
    return results

def api_handle_sar(data, classification, is_changed):
    ret_json = sar(data['input_image']['image_url'])
    # print(f"Ret JSON: {ret_json}")
    response_json = {}
    response_json['input_image'] = data['input_image']
    response_json['queries'] = {}
    response_json['queries']['caption_query'] = {}
    response_json['queries']['caption_query']['instruction'] = data['queries']['caption_query']['instruction']
    response_json['queries']['caption_query']['response'] = ret_json['caption']
    response_json['queries']['grounding_query'] = {}
    response_json['queries']['grounding_query']['instruction'] = data['queries']['grounding_query']['instruction']
    response_json['queries']['grounding_query']['response'] = grounding(data['input_image']['image_url'], data['queries']['grounding_query']['instruction'] + " " + ret_json['llama_guide_text'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    response_json['queries']['attribute_query'] = {}
    response_json['queries']['attribute_query']['binary'] = {}
    response_json['queries']['attribute_query']['binary']['instruction'] = data['queries']['attribute_query']['binary']['instruction']
    response_json['queries']['attribute_query']['binary']['response'] = binary_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['binary']['instruction'] + " " + ret_json['llama_guide_text'])
    response_json['queries']['attribute_query']['numeric'] = {}
    response_json['queries']['attribute_query']['numeric']['instruction'] = data['queries']['attribute_query']['numeric']['instruction']
    response_json['queries']['attribute_query']['numeric']['response'] = numeric_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['numeric']['instruction'] + " " + ret_json['llama_guide_text'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    response_json['queries']['attribute_query']['semantic'] = {}
    response_json['queries']['attribute_query']['semantic']['instruction'] = data['queries']['attribute_query']['semantic']['instruction']
    response_json['queries']['attribute_query']['semantic']['response'] = semantic_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['semantic']['instruction'] + " " + ret_json['llama_guide_text'])

    if is_changed:
        # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
        last_part = data['input_image']['image_url'].split('/')[-1]
        new_url = f"{last_part}"
        # if new_url ends in ? remove it
        if new_url.endswith('?'):
            new_url = new_url[:-1]
        supabase.storage.from_(BUCKET_NAME).remove([new_url])
    # response_json['input_image'] = data['input_image']
    # response_json['queries'] = {}
    # response_json['queries']['caption_query'] = {}
    # response_json['queries']['caption_query']['instruction'] = data['queries']['caption_query']['instruction']
    # response_json['queries']['caption_query']['response'] = caption(data['input_image']['image_url'], data['queries']['caption_query']['instruction'])
    # response_json['queries']['grounding_query'] = {}
    # response_json['queries']['grounding_query']['instruction'] = data['queries']['grounding_query']['instruction']
    # response_json['queries']['grounding_query']['response'] = grounding(data['input_image']['image_url'], data['queries']['grounding_query']['instruction'], classification=classification)
    # response_json['queries']['attribute_query'] = {}
    # response_json['queries']['attribute_query']['binary'] = {}
    # response_json['queries']['attribute_query']['binary']['instruction'] = data['queries']['attribute_query']['binary']['instruction']
    # response_json['queries']['attribute_query']['binary']['response'] = binary_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['binary']['instruction'])
    # response_json['queries']['attribute_query']['numeric'] = {}
    # response_json['queries']['attribute_query']['numeric']['instruction'] = data['queries']['attribute_query']['numeric']['instruction']
    # response_json['queries']['attribute_query']['numeric']['response'] = numeric_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['numeric']['instruction'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    # response_json['queries']['attribute_query']['semantic'] = {}
    # response_json['queries']['attribute_query']['semantic']['instruction'] = data['queries']['attribute_query']['semantic']['instruction']
    # response_json['queries']['attribute_query']['semantic']['response'] = semantic_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['semantic']['instruction'])

    return jsonify(response_json), 200

def ui_sar_caption(img_url, prompt):
    ret_json = sar(img_url)
    return ret_json['caption']

def ui_fcc_caption(img_url, prompt):
    ret_json = fcc(img_url)
    return ret_json['caption']

def ui_sar_grounding(img_url, prompt, classification):
    ret_json = sar(img_url)
    results = grounding(img_url, prompt + " " + ret_json['llama_guide_text'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    return results

def ui_fcc_grounding(img_url, prompt, classification):
    #for fcc, I will convert it into natural color composite (NCC)
    #fcc_to_ncc(img_url) will return url of ncc image to be used ahead.
    #then I will use the ncc image for grounding
    ncc_image_url = fcc_to_ncc(img_url)
    results = grounding(ncc_image_url, prompt, yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    return results

def ui_sar_numeric_vqa(img_url, prompt, classification):
    ret_json = sar(img_url)
    results = numeric_vqa(img_url, prompt + " " + ret_json['llama_guide_text'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    
    return results

def ui_fcc_numeric_vqa(img_url, prompt, classification):
    #for fcc, I will convert it into natural color composite (NCC)
    #fcc_to_ncc(img_url) will return url of ncc image to be used ahead.
    #then I will use the ncc image for numeric vqa
    ncc_image_url = fcc_to_ncc(img_url)
    results = numeric_vqa(ncc_image_url, prompt, yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    return results

def ui_sar_semantic_vqa(img_url, prompt, classification):
    ret_json = sar(img_url)
    results = semantic_vqa(img_url, prompt + " " + ret_json['llama_guide_text'])
    return results

def ui_fcc_semantic_vqa(img_url, prompt, classification):
    #for fcc, I will convert it into natural color composite (NCC)
    #fcc_to_ncc(img_url) will return url of ncc image to be used ahead.
    #then I will use the ncc image for semantic vqa
    ncc_image_url = fcc_to_ncc(img_url)
    results = semantic_vqa(ncc_image_url, prompt)
    return results

def ui_sar_binary_vqa(img_url, prompt, classification):
    ret_json = sar(img_url)
    results = binary_vqa(img_url, prompt + " " + ret_json['llama_guide_text'])
    return results

def ui_fcc_binary_vqa(img_url, prompt, classification):
    #for fcc, I will convert it into natural color composite (NCC)
    #fcc_to_ncc(img_url) will return url of ncc image to be used ahead.
    #then I will use the ncc image for binary vqa
    ncc_image_url = fcc_to_ncc(img_url)
    results = binary_vqa(ncc_image_url, prompt)
    return results

def ui_sar_general_vqa(messages):
    ret_json = sar(messages[-1]['content'][0]['image'])
    results = general_vqa(messages + " " + ret_json['llama_guide_text'])
    return results

def ui_fcc_general_vqa(messages):
    #for fcc, I will convert it into natural color composite (NCC)
    #fcc_to_ncc(img_url) will return url of ncc image to be used ahead.
    #then I will use the ncc image for general vqa
    ncc_image_url = fcc_to_ncc(messages[-1]['content'][0]['image'])
    results = general_vqa(messages)
    return results

def api_handle_fcc(data, classification, is_changed):
    ret_json = fcc(data['input_image']['image_url'])
    print(f"Ret JSON: {ret_json}")
    response_json = {}
    response_json['input_image'] = data['input_image']
    response_json['queries'] = {}
    response_json['queries']['caption_query'] = {}
    response_json['queries']['caption_query']['instruction'] = data['queries']['caption_query']['instruction']
    response_json['queries']['caption_query']['response'] = ret_json['caption']
    #for grounding and vqa, I will convert it into natural color composite (NCC)
    #fcc_to_ncc(img_url) will return url of ncc image to be used ahead.
    #then I will use the ncc image for grounding and vqa
    ncc_image_url = fcc_to_ncc(data['input_image']['image_url'])
    # return jsonify({"ncc_image_url": ncc_image_url}), 200
    response_json['queries']['grounding_query'] = {}
    response_json['queries']['grounding_query']['instruction'] = data['queries']['grounding_query']['instruction']
    response_json['queries']['grounding_query']['response'] = grounding(ncc_image_url, data['queries']['grounding_query']['instruction'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    response_json['queries']['attribute_query'] = {}
    response_json['queries']['attribute_query']['binary'] = {}
    response_json['queries']['attribute_query']['binary']['instruction'] = data['queries']['attribute_query']['binary']['instruction']
    response_json['queries']['attribute_query']['binary']['response'] = binary_vqa(ncc_image_url, data['queries']['attribute_query']['binary']['instruction'])
    response_json['queries']['attribute_query']['numeric'] = {}
    response_json['queries']['attribute_query']['numeric']['instruction'] = data['queries']['attribute_query']['numeric']['instruction']
    response_json['queries']['attribute_query']['numeric']['response'] = numeric_vqa(ncc_image_url, data['queries']['attribute_query']['numeric']['instruction'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
    response_json['queries']['attribute_query']['semantic'] = {}
    response_json['queries']['attribute_query']['semantic']['instruction'] = data['queries']['attribute_query']['semantic']['instruction']
    response_json['queries']['attribute_query']['semantic']['response'] = semantic_vqa(ncc_image_url, data['queries']['attribute_query']['semantic']['instruction'])

    if is_changed:
        # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
        last_part = data['input_image']['image_url'].split('/')[-1]
        new_url = f"{last_part}"
        # if new_url ends in ? remove it
        if new_url.endswith('?'):
            new_url = new_url[:-1]
        supabase.storage.from_(BUCKET_NAME).remove([new_url])
    return jsonify(response_json), 200

def reduce_image_size(image_url: str) -> Tuple[str, float]:
    """
    Downloads an image, checks its dimensions. 
    If max(width, height) > THRESHOLD, resizes it preserving aspect ratio,
    uploads to Supabase, and returns (new_url, scaling_factor).
    
    Otherwise, returns (original_url, 1.0).
    """
    try:
        THRESHOLD = 750
        # 1. Download the image
        print(f"Fetching image: {image_url}")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Load image into Pillow
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Capture original dimensions
        original_width, original_height = img.size
        print(f"Original Dimensions: {original_width}x{original_height}")

        # 2. Check dimensions
        if max(original_width, original_height) <= THRESHOLD:
            print("Image is within threshold. Returning original URL.")
            # No resizing happened, so scaling factor is 1.0
            return image_url, 1.0

        # 3. Resize preserving Aspect Ratio
        img.thumbnail((THRESHOLD, THRESHOLD))
        
        new_width, new_height = img.size
        print(f"Resized Dimensions: {new_width}x{new_height}")

        # CALCULATE SCALING FACTOR
        # We can use width or height; since aspect ratio is preserved, the ratio is the same.
        scaling_factor = new_width / original_width

        # 4. Save resized image to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        file_bytes = buffer.getvalue()
        
        # Generate a unique filename
        file_ext = "jpg"
        file_name = f"resized_{uuid.uuid4()}.{file_ext}"

        # 5. Upload to Supabase
        print("Uploading resized image to Supabase...")
        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_name, 
            file=file_bytes, 
            file_options={"content-type": "image/jpeg"}
        )

        # 6. Retrieve Public URL
        public_url_response = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
        new_url = public_url_response
        
        print(f"New URL generated: {new_url} | Factor: {scaling_factor}")
        
        return new_url, scaling_factor

    except Exception as e:
        print(f"Error processing image: {e}")
        # In case of error, fallback to original URL. 
        # Treat factor as 1.0 (no change) so downstream math doesn't break.
        return image_url, 1.0

def transform_numeric_answer(answer: float, classification: QueryClassification, scaling_factor: float, gsd: float) -> float:
    if classification.vqa_float_subtype == "area":
        return answer/(scaling_factor**2)*gsd**2
    elif classification.vqa_float_subtype == "perimeter":
        return answer/(scaling_factor)*gsd
    elif classification.vqa_float_subtype == "length":
        return answer/(scaling_factor)*gsd
    else:
        return answer

@app.route('/api', methods=['POST'])
def api():
    """
    Process a query JSON like sample1_query.json

    Returns:
        JSON response like sample1_response.json
    """
    data = request.json
    print(data)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    messages_ground = [{"role": "user", "content":[
        {"type": "image", "image": data['input_image']['image_url']},
        {"type": "text", "text": data['queries']['grounding_query']['instruction']}
    ]}]

    messages_numeric = [{"role": "user", "content":[
        {"type": "image", "image": data['input_image']['image_url']},
        {"type": "text", "text": data['queries']['attribute_query']['numeric']['instruction']}
    ]}]
    
    image_class = image_classify(data['input_image']['image_url'], best_model)
    print(f"Image class: {image_class}")
    classification_grounding = classify_query_from_messages(messages_ground)
    classification_numeric = classify_query_from_messages(messages_numeric)
    # classification = QueryClassification(task_type="grounding", grounding_subtype="multiple", object_class="swimming pool")
    response_json = {}

    # image_class = "SAR"

    is_changed = False
    new_url, scaling_factor = reduce_image_size(data['input_image']['image_url'])
    gsd = data['input_image']['metadata']['spatial_resolution_m']
    
    if new_url != data['input_image']['image_url']:
        is_changed = True
    data['input_image']['image_url'] = new_url

    # if image_class == "SAR":
    #     print("SAR detected")
    #     return api_handle_sar(data, classification, is_changed)
    # elif image_class == "FCC":
    #     print("FCC detected")
    #     return api_handle_fcc(data, classification, is_changed)

    if image_class == "FCC":
        new_url = fcc_to_ncc(data['input_image']['image_url'])
        data['input_image']['image_url'] = new_url




    response_json['input_image'] = data['input_image']
    response_json['queries'] = {}
    response_json['queries']['caption_query'] = {}
    response_json['queries']['caption_query']['instruction'] = data['queries']['caption_query']['instruction']
    response_json['queries']['caption_query']['response'] = caption(data['input_image']['image_url'], data['queries']['caption_query']['instruction'])
    response_json['queries']['grounding_query'] = {}
    response_json['queries']['grounding_query']['instruction'] = data['queries']['grounding_query']['instruction']
    response_json['queries']['grounding_query']['response'] = convert_obb_to_point_representation(grounding(data['input_image']['image_url'], data['queries']['grounding_query']['instruction'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification_grounding))
    # response_json['queries']['grounding_query']['response'] = grounding(data['input_image']['image_url'], data['queries']['grounding_query']['instruction'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification_grounding)
    response_json['queries']['attribute_query'] = {}
    response_json['queries']['attribute_query']['binary'] = {}
    response_json['queries']['attribute_query']['binary']['instruction'] = data['queries']['attribute_query']['binary']['instruction']
    response_json['queries']['attribute_query']['binary']['response'] = binary_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['binary']['instruction'])
    response_json['queries']['attribute_query']['numeric'] = {}
    response_json['queries']['attribute_query']['numeric']['instruction'] = data['queries']['attribute_query']['numeric']['instruction']
    # response_json['queries']['attribute_query']['numeric']['response'] = numeric_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['numeric']['instruction'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification_numeric)
    numeric_answer = numeric_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['numeric']['instruction'], yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification_numeric)
    numeric_answer = transform_numeric_answer(numeric_answer, classification_numeric, scaling_factor, gsd)
    response_json['queries']['attribute_query']['numeric']['response'] = numeric_answer
    response_json['queries']['attribute_query']['semantic'] = {}
    response_json['queries']['attribute_query']['semantic']['instruction'] = data['queries']['attribute_query']['semantic']['instruction']
    response_json['queries']['attribute_query']['semantic']['response'] = semantic_vqa(data['input_image']['image_url'], data['queries']['attribute_query']['semantic']['instruction'])

    if is_changed:
        # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
        try:
            last_part = data['input_image']['image_url'].split('/')[-1]
            new_url = f"{last_part}"
            # if new_url ends in ? remove it
            if new_url.endswith('?'):
                new_url = new_url[:-1]
            supabase.storage.from_(BUCKET_NAME).remove([new_url])
        except Exception as e:
            print(f"Error removing image: {e}")
    return jsonify(response_json), 200

@app.route('/imageclass', methods=['POST'])
def imageclass():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    image_class = image_classify(data['image_url'], best_model)
    if image_class == "NCC":
        image_class = "Optical"
    return jsonify({'image_class': image_class}), 200

@app.route('/analyze', methods=['POST'])
def ui_chat():
    """
    Endpoint for UI chat interface.
    
    Accepts messages array either as:
    - Query parameter: ?messages=[{"role":"user","content":"..."}]
    - JSON body: {"messages": [{"role":"user","content":"..."}]}
    """
    import json
    
    messages = None

    print(request.args)
    
    # Try to get messages from query parameter first
    if 'conversation' in request.args:
        try:
            messages_str = request.args.get('conversation')
            messages = json.loads(messages_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON in messages query parameter: {str(e)}'}), 400
    
    # If not in query params, try request body
    elif request.is_json:
        data = request.get_json()
        if data and 'conversation' in data:
            messages = data['conversation']
    
    # If still not found, return error
    if messages is None:
        return jsonify({'error': 'messages array not found in query parameters or request body'}), 400
    
    # Validate messages is a list
    if not isinstance(messages, list):
        return jsonify({'error': 'messages must be an array'}), 400
    
    print(f"Received messages: {messages}")
    
    # Helper function to extract image_url and prompt from messages content array
    def extract_image_and_text(content_array):
        """Extract image URL and text prompt from content array where items may be in any order."""
        image_url = None
        prompt = None
        for item in content_array:
            if isinstance(item, dict):
                if 'image' in item:
                    image_url = item['image']
                elif 'text' in item:
                    prompt = item['text']
        return image_url, prompt


    new_url, scaling_factor = reduce_image_size(messages[-1]['content'][0]['image'])
    gsd = 1
    for objs in messages[-1]['content']:
        if objs['type'] == 'dimensions':
            gsd = objs['height']
            break
    is_changed = False
    if new_url != messages[-1]['content'][0]['image']:
        is_changed = True
    messages[-1]['content'][0]['image'] = new_url

    # Classify the query from messages
    try:
        classification = classify_query_from_messages(messages)

        img_class = image_classify(messages[-1]['content'][0]['image'], best_model)
        print(f"Image class: {img_class}")

        # image_url, prompt = extract_image_and_text(messages[-1]['content'])
        # image_class = image_classify(image_url)
        # print(f"Image classification: {image_class}")
        # return jsonify({"answer": image_class}), 200
        # classification = QueryClassification(task_type="grounding", grounding_subtype="multiple", object_class="swimming pool")
        print(f"Classification: {classification}")
        print(classification.task_type)
        if classification.task_type == "captioning":
            image_url, prompt = extract_image_and_text(messages[-1]['content'])
            if img_class == "SAR":
                gen_caption = ui_sar_caption(image_url, prompt)
            if img_class == "FCC":
                image_url = fcc_to_ncc(image_url)
                messages[-1]['content'][0]['image'] = image_url
            gen_caption = caption(image_url, prompt)
            # gen_caption = "This is a caption"

            if is_changed:
                # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
                last_part = image_url.split('/')[-1]
                new_url = f"{last_part}"
                # if new_url ends in ? remove it
                if new_url.endswith('?'):
                    new_url = new_url[:-1]
                supabase.storage.from_(BUCKET_NAME).remove([new_url])
            return jsonify({"answer": gen_caption}), 200
        elif classification.task_type == "grounding":
            print(messages)
            image_url, prompt = extract_image_and_text(messages[-1]['content'])
            if img_class == "SAR":
                gen_obbs = ui_sar_grounding(image_url, prompt, classification)
            if img_class == "FCC":
                image_url = fcc_to_ncc(image_url)
                messages[-1]['content'][0]['image'] = image_url
            gen_obbs = grounding(image_url, prompt, yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
            class_name = classification.object_class
            response_json = {}
            response_json['answer'] = ("The bounding boxes are drawn in the image." if gen_obbs else "No bounding boxes found in the image.")

            # Load image to get dimensions for converting normalized OBB to absolute values
            try:
                response = requests.get(image_url)
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                img_width, img_height = image.size
                
            except Exception as e:
                LOGGER.error(f"Failed to load image to get dimensions: {e}")
                # Fallback: use default dimensions (shouldn't happen, but handle gracefully)
                img_width, img_height = 512, 512

            obbs_array = []
            for obb in gen_obbs:
                # Convert normalized OBB values (0-1) to absolute pixel values
                cx_norm, cy_norm, w_norm, h_norm, angle_cv2 = obb["obbox"]
                cx_abs = cx_norm * img_width
                cy_abs = cy_norm * img_height
                w_abs = w_norm * img_width
                h_abs = h_norm * img_height
                
                # Convert angle from cv2.minAreaRect format ([-90, 0) degrees) to frontend format ([0, 180) degrees)
                angle_frontend = convert_cv2_angle_to_frontend_format(angle_cv2)
                
                json_obb = {
                    "id": obb["object-id"],
                    "category": class_name,
                    "center":{
                        "x": cx_abs,
                        "y": cy_abs,
                    },
                    "size":{
                        "width": w_abs,
                        "height": h_abs,
                    },
                    "angle": angle_frontend,
                }
                obbs_array.append(json_obb)
            response_json['groundingData'] = [{
                class_name: obbs_array
            }]

            if is_changed:
                # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
                last_part = image_url.split('/')[-1]
                new_url = f"{last_part}"
                # if new_url ends in ? remove it
                if new_url.endswith('?'):
                    new_url = new_url[:-1]
                supabase.storage.from_(BUCKET_NAME).remove([new_url])
            return jsonify(response_json), 200
        elif classification.task_type == "vqa_float":
            image_url, prompt = extract_image_and_text(messages[-1]['content'])
            if img_class == "SAR":
                gen_answer = ui_sar_numeric_vqa(image_url, prompt, classification)
            if img_class == "FCC":
                image_url = fcc_to_ncc(image_url)
                messages[-1]['content'][0]['image'] = image_url
            gen_answer = numeric_vqa(image_url, prompt, yolo_model_1=runs_obb_model, yolo_model_2=last_model, yolo_base=base_yolo, classification=classification)
            gen_answer = transform_numeric_answer(gen_answer, classification, scaling_factor, gsd)
            # response_json['queries']['attribute_query']['numeric']['response'] = gen_answer
            if is_changed:
                # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
                last_part = image_url.split('/')[-1]
                new_url = f"{last_part}"
                # if new_url ends in ? remove it
                if new_url.endswith('?'):
                    new_url = new_url[:-1]
                supabase.storage.from_(BUCKET_NAME).remove([new_url])
            return jsonify({"answer": str(gen_answer)}), 200
        elif classification.task_type == "semantic":
            image_url, prompt = extract_image_and_text(messages[-1]['content'])
            if img_class == "SAR":
                gen_answer = ui_sar_semantic_vqa(image_url, prompt, classification)
            if img_class == "FCC":
                image_url = fcc_to_ncc(image_url)
                messages[-1]['content'][0]['image'] = image_url
            gen_answer = semantic_vqa(image_url, prompt)

            if is_changed:
                # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
                last_part = image_url.split('/')[-1]
                new_url = f"{last_part}"
                # if new_url ends in ? remove it
                if new_url.endswith('?'):
                    new_url = new_url[:-1]
                supabase.storage.from_(BUCKET_NAME).remove([new_url])
            return jsonify({"answer": gen_answer}), 200
        elif classification.task_type == "binary":
            image_url, prompt = extract_image_and_text(messages[-1]['content'])
            if img_class == "SAR":
                gen_answer = ui_sar_binary_vqa(image_url, prompt, classification)
            if img_class == "FCC":
                image_url = fcc_to_ncc(image_url)
                messages[-1]['content'][0]['image'] = image_url
            gen_answer = binary_vqa(image_url, prompt)

            if is_changed:
                # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
                last_part = image_url.split('/')[-1]
                new_url = f"{last_part}"
                # if new_url ends in ? remove it
                if new_url.endswith('?'):
                    new_url = new_url[:-1]
                supabase.storage.from_(BUCKET_NAME).remove([new_url])
            return jsonify({"answer": gen_answer}), 200
        elif classification.task_type == "general":
            # image_url, prompt = extract_image_and_text(messages[-1]['content'])
            image_url = messages[-1]['content'][0]['image']
            if img_class == "SAR":
                gen_answer = ui_sar_general_vqa(messages)
            if img_class == "FCC":
                image_url = fcc_to_ncc(image_url)
                messages[-1]['content'][0]['image'] = image_url
            gen_answer = general_vqa(messages)

            if is_changed:
                # process the current url so to get the last part of the url (divided by /) and add an image/ prefix to it
                last_part = image_url.split('/')[-1]
                new_url = f"{last_part}"
                # if new_url ends in ? remove it
                if new_url.endswith('?'):
                    new_url = new_url[:-1]
                supabase.storage.from_(BUCKET_NAME).remove([new_url])
            return jsonify({"answer": gen_answer}), 200
    except Exception as e:
        LOGGER.error(f"Error classifying query: {str(e)}")
        return jsonify({'error': f'Failed to classify query: {str(e)}'}), 500

@app.route('/agriculture', methods=['POST'])
def agriculture():
    """
    Endpoint for agriculture bot inference.
    Accepts messages array either as:
    - Query parameter: ?messages=[{"role":"user","content":"..."}]
    - JSON body: {"messages": [{"role":"user","content":"..."}]}
    Returns:
        JSON response with 'answer' and optional 'graph' fields
    """
    import json
    messages = None

    # Try to get messages from query parameter first
    print(request.args) 
    if 'conversation' in request.args:
        try:
            messages_str = request.args.get('conversation')
            messages = json.loads(messages_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON in messages query parameter: {str(e)}'}), 400

    # If not in query params, try request body
    elif request.is_json:
        data = request.get_json()
        if data and 'conversation' in data:
            messages = data['conversation']

    if messages is None:
        return jsonify({'error': 'conversation array not found in query parameters or request body', 'answer': "Please provide a conversation to analyze."}), 400

    # Validate messages is a list
    if not isinstance(messages, list):
        return jsonify({'error': 'conversation must be an array', 'answer': "Please provide a conversation to analyze."}), 400

    print(messages)

    print(f"Received conversation: {messages[-1]}")

    # Helper function to extract image_url and question from messages content array

    def extract_image_and_text(content_array):
        """Extract image URL and text prompt from content array where items may be in any order."""
        image_url = None
        question = None
        for item in content_array:
            if 'image' in item:
                image_url = item['image']
            elif 'text' in item:
                question = item['text']
        return image_url, question

    try:
        image_url, question = extract_image_and_text(messages[-1]["content"])
        if image_url is None:
            return jsonify({'error': 'image URL not found in messages', 'answer': "Please provide an image to analyze."}), 400

        if question is None:
            return jsonify({'error': 'question/text not found in messages', 'answer': "Please provide a question/text to analyze the image."}), 400
        result = get_model_response(image_url, question)
        print(f"Result: {result}")
        return jsonify(result), 200
    except Exception as e:
        LOGGER.error(f"Error in agriculture endpoint: {str(e)}")
        return jsonify({'error': f'Failed to process agriculture query: {str(e)}', 'answer': "Some error occurred. Please try again later."}), 500

@app.route('/tgcompare', methods=['POST'])
def compare():
    import json
    messages = None

    # Try to get messages from query parameter first
    print(request.args) 
    if 'conversation' in request.args:
        try:
            messages_str = request.args.get('conversation')
            messages = json.loads(messages_str)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON in messages query parameter: {str(e)}'}), 400

    # If not in query params, try request body
    elif request.is_json:
        data = request.get_json()
        if data and 'conversation' in data:
            messages = data['conversation']

    if messages is None:
        return jsonify({'error': 'conversation array not found in query parameters or request body', 'answer': "Please provide a conversation to analyze."}), 400

    # Validate messages is a list
    if not isinstance(messages, list):
        return jsonify({'error': 'conversation must be an array', 'answer': "Please provide a conversation to analyze."}), 400

    final_messages = messages[-1]
    final_messages['content'][0]['type'] = 'image'
    final_messages['content'][1]['type'] = 'image'

    result = qwen_base([final_messages])

    return jsonify({'answer': result}), 200



@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with pipeline status."""
    return jsonify({
        "status": "ok"
    }), 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
