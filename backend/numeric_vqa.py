"""
Numeric VQA module with counting ensemble support.

This module handles numeric VQA queries (counting and size measurements) using:
- Counting queries: YOLO ensemble + base Qwen model (via modal)
- Size queries: Base Qwen model only (via modal)
"""

import logging
import re
import requests
import io
import base64
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.distance import cdist

from ultralytics import YOLO
YOLO_AVAILABLE = True

from modal_request import qwen_base, sam_ensemble_mask
from overall_classifier import classify_query
from grounding import grounding

LOGGER = logging.getLogger(__name__)


def _parse_numeric_answer(text: str) -> float | str:
    """Parse numeric answer from text, similar to tasks.py"""
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return text.strip()
    return text.strip()


# Known object classes for VRSBench-obb (matching overall_classifier.py)
KNOWN_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
    "windmill", "dam", "trainstation", "overpass", "stadium", "airport", "helipad",
    "golffield", "chimney", "expressway service area", "expressway toll station",
    "container crane"
]

# YOLO class names (all classes from KNOWN_CLASSES in Title Case format)
YOLO_CLASSES = [
    "Plane",
    "Ship",
    "Storage Tank",
    "Baseball Diamond",
    "Tennis Court",
    "Basketball Court",
    "Ground Track Field",
    "Harbor",
    "Bridge",
    "Large Vehicle",
    "Small Vehicle",
    "Helicopter",
    "Roundabout",
    "Soccer Ball Field",
    "Swimming Pool",
    "Windmill",
    "Dam",
    "Trainstation",
    "Overpass",
    "Stadium",
    "Airport",
    "Helipad",
    "Golffield",
    "Chimney",
    "Expressway Service Area",
    "Expressway Toll Station",
    "Container Crane",
]

# Class name normalization mapping
CLASS_NAME_MAPPING = {
    # Plane variations
    "airplane": "Plane",
    "aircraft": "Plane",
    "plane": "Plane",
    # Ship variations
    "ship": "Ship",
    "vessel": "Ship",
    "boat": "Ship",
    # Storage Tank variations
    "storagetank": "Storage Tank",
    "storage tank": "Storage Tank",
    "storage-tank": "Storage Tank",
    # Baseball Diamond variations
    "baseballfield": "Baseball Diamond",
    "baseball diamond": "Baseball Diamond",
    "baseball-diamond": "Baseball Diamond",
    "baseball field": "Baseball Diamond",
    # Tennis Court variations
    "tenniscourt": "Tennis Court",
    "tennis court": "Tennis Court",
    "tennis-court": "Tennis Court",
    # Basketball Court variations
    "basketballcourt": "Basketball Court",
    "basketball court": "Basketball Court",
    "basketball-court": "Basketball Court",
    # Ground Track Field variations
    "groundtrackfield": "Ground Track Field",
    "ground track field": "Ground Track Field",
    "ground-track-field": "Ground Track Field",
    "track field": "Ground Track Field",
    "running track": "Ground Track Field",
    # Harbor variations
    "harbor": "Harbor",
    "harbour": "Harbor",
    "port": "Harbor",
    # Bridge
    "bridge": "Bridge",
    # Vehicle variations
    "vehicle": "Small Vehicle",  # Default to small vehicle
    "car": "Small Vehicle",
    "automobile": "Small Vehicle",
    "truck": "Large Vehicle",
    "bus": "Large Vehicle",
    "large vehicle": "Large Vehicle",
    "large-vehicle": "Large Vehicle",
    "small vehicle": "Small Vehicle",
    "small-vehicle": "Small Vehicle",
    # Helicopter variations
    "helicopter": "Helicopter",
    "chopper": "Helicopter",
    # Roundabout variations
    "roundabout": "Roundabout",
    "traffic circle": "Roundabout",
    # Soccer Ball Field variations
    "soccer ball field": "Soccer Ball Field",
    "soccer-ball-field": "Soccer Ball Field",
    "soccer field": "Soccer Ball Field",
    "football field": "Soccer Ball Field",
    # Swimming Pool variations
    "swimming pool": "Swimming Pool",
    "swimming-pool": "Swimming Pool",
    "pool": "Swimming Pool",
    # Windmill
    "windmill": "Windmill",
    # Dam
    "dam": "Dam",
    # Trainstation variations
    "trainstation": "Trainstation",
    "train station": "Trainstation",
    "train-station": "Trainstation",
    # Overpass
    "overpass": "Overpass",
    # Stadium
    "stadium": "Stadium",
    # Airport
    "airport": "Airport",
    # Helipad
    "helipad": "Helipad",
    "heli pad": "Helipad",
    "heli-pad": "Helipad",
    # Golffield variations
    "golffield": "Golffield",
    "golf field": "Golffield",
    "golf-field": "Golffield",
    # Chimney
    "chimney": "Chimney",
    # Expressway Service Area variations
    "expressway service area": "Expressway Service Area",
    "expressway-service-area": "Expressway Service Area",
    # Expressway Toll Station variations
    "expressway toll station": "Expressway Toll Station",
    "expressway-toll-station": "Expressway Toll Station",
    # Container Crane variations
    "container crane": "Container Crane",
    "container-crane": "Container Crane",
    "crane": "Container Crane",
}


def normalize_class_name(class_name: str) -> str:
    """
    Normalize class name to match YOLO class names.
    
    Args:
        class_name: Class name from classification (may have different format)
    
    Returns:
        Normalized class name that matches one of the YOLO class names, or original if no match
    """
    if not class_name:
        return class_name
    
    # Check direct mapping first
    if class_name in CLASS_NAME_MAPPING:
        return CLASS_NAME_MAPPING[class_name]
    
    # Normalize the input: lowercase, replace hyphens with spaces
    normalized_input = class_name.lower().replace("-", " ").replace("_", " ").strip()
    normalized_input = re.sub(r'\s+', ' ', normalized_input)
    
    # Check normalized input in mapping
    if normalized_input in CLASS_NAME_MAPPING:
        return CLASS_NAME_MAPPING[normalized_input]
    
    # Try exact match against YOLO classes (case-insensitive)
    for yolo_name in YOLO_CLASSES:
        if yolo_name.lower() == normalized_input:
            return yolo_name
    
    # Try matching word by word
    input_words = set(normalized_input.split())
    for yolo_name in YOLO_CLASSES:
        yolo_words = set(yolo_name.lower().split())
        if input_words == yolo_words:
            return yolo_name
    
    # Return original if no match found
    LOGGER.warning(f"Could not normalize class name: {class_name}, using as-is")
    return class_name


def extract_count(answer: str) -> Optional[int]:
    """
    Robust integer extraction from answer text.
    Accepts numeric string, float convertible, first number in text, or textual numbers.
    """
    if answer is None:
        return None
    answer = str(answer).strip()
    
    # Direct int/float
    try:
        return int(round(float(answer)))
    except ValueError:
        pass
    
    # Regex number
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if nums:
        try:
            return int(round(float(nums[0])))
        except ValueError:
            pass
    
    # Textual numbers
    mapping = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12
    }
    lower = answer.lower()
    for word, val in mapping.items():
        if word in lower:
            return val
    return None


class YOLOCounter:
    """Simple YOLO counter for counting objects by class name."""
    def __init__(self, model):
        self.model = model
        # Get class names map
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            self.class_names_map: Dict[int, str] = model.model.names
            self.class_names_list: List[str] = list(self.class_names_map.values())
        elif hasattr(model, 'names'):
            self.class_names_map: Dict[int, str] = model.names
            self.class_names_list: List[str] = list(self.class_names_map.values())
        else:
            self.class_names_map = {}
            self.class_names_list = []

    def get_class_names(self) -> List[str]:
        return self.class_names_list

    def predict(self, image: Image.Image, conf: float = 0.25, iou: float = 0.5):
        return self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False
        )

    def count_class(self, results, target_class_name: str) -> int:
        """Count objects of a specific class from YOLO results."""
        total = 0
        LOGGER.info(f"Target class name: {target_class_name}")
        LOGGER.info(f"Available class names in model: {self.class_names_list}")
        LOGGER.info(f"Class names map: {self.class_names_map}")
        
        # Try to find the best matching class name from the model's actual class names
        # This handles cases where the normalized name doesn't exactly match model class names
        target_lower = target_class_name.lower().strip()
        matching_model_class = None
        
        # First, try exact case-insensitive match
        for model_class in self.class_names_list:
            if model_class.lower().strip() == target_lower:
                matching_model_class = model_class
                LOGGER.info(f"Found exact match: '{target_class_name}' -> '{matching_model_class}'")
                break
        
        # If no exact match, try word-by-word matching
        if matching_model_class is None:
            target_words = set(target_lower.split())
            for model_class in self.class_names_list:
                model_words = set(model_class.lower().strip().split())
                if target_words == model_words:
                    matching_model_class = model_class
                    LOGGER.info(f"Found word match: '{target_class_name}' -> '{matching_model_class}'")
                    break
        
        # If still no match, try partial matching (target words subset of model words or vice versa)
        if matching_model_class is None:
            target_words = set(target_lower.split())
            for model_class in self.class_names_list:
                model_words = set(model_class.lower().strip().split())
                if target_words.issubset(model_words) or model_words.issubset(target_words):
                    matching_model_class = model_class
                    LOGGER.info(f"Found partial match: '{target_class_name}' -> '{matching_model_class}'")
                    break
        
        # Use the matching class name if found, otherwise use original target
        search_class = matching_model_class if matching_model_class else target_class_name
        if matching_model_class is None:
            LOGGER.warning(f"No matching class found in model for '{target_class_name}', will use case-insensitive comparison")
        
        all_detected_classes = []
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None and r.boxes.cls is not None:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for cid in cls_ids:
                    cname = self.class_names_map.get(int(cid), str(cid))
                    all_detected_classes.append(cname)
                    # Compare with both the search class and original target
                    if (cname.lower().strip() == search_class.lower().strip() or 
                        cname.lower().strip() == target_class_name.lower().strip()):
                        total += 1
            # Handle OBB if present
            if hasattr(r, "obb") and r.obb is not None and r.obb.cls is not None:
                cls_ids_obb = r.obb.cls.cpu().numpy().astype(int)
                for cid in cls_ids_obb:
                    cname = self.class_names_map.get(int(cid), str(cid))
                    all_detected_classes.append(cname)
                    # Compare with both the search class and original target
                    if (cname.lower().strip() == search_class.lower().strip() or 
                        cname.lower().strip() == target_class_name.lower().strip()):
                        total += 1
        
        if all_detected_classes:
            from collections import Counter
            detected_counts = Counter(all_detected_classes)
            LOGGER.info(f"All detected classes: {dict(detected_counts)}")
        else:
            LOGGER.warning("No detections found in results (check confidence threshold)")
        
        LOGGER.info(f"Final count for '{target_class_name}': {total}")
        return int(total)


class EnsembleYOLOCounter:
    """
    Ensemble YOLO counter that uses primary model first, then falls back to secondary model,
    and finally to base model if both return 0 detections.
    """
    def __init__(self, yolo_model_1, yolo_model_2, yolo_base=None):
        """
        Initialize ensemble with two or three YOLO models.
        
        Args:
            yolo_model_1: Primary YOLO model (used first)
            yolo_model_2: Secondary YOLO model (fallback if primary finds nothing)
            yolo_base: Base YOLO model (final fallback, defaults to yolo11l-obb.pt)
        """
        LOGGER.info("Initializing YOLO ensemble counter")
        self.primary_counter = YOLOCounter(yolo_model_1)
        self.secondary_counter = YOLOCounter(yolo_model_2)
        
        # Initialize base counter if provided
        self.base_counter = None
        if yolo_base is not None:
            self.base_counter = YOLOCounter(yolo_base)
        
        # Merge class names from all models (primary takes precedence)
        primary_classes = set(self.primary_counter.get_class_names())
        secondary_classes = set(self.secondary_counter.get_class_names())
        base_classes = set(self.base_counter.get_class_names()) if self.base_counter else set()
        self.class_names_list = list(primary_classes.union(secondary_classes).union(base_classes))

    def get_class_names(self) -> List[str]:
        """Get all unique class names from both models."""
        return self.class_names_list

    def predict_and_count(
        self, 
        image: Image.Image, 
        target_class_name: str,
        conf: float = 0.25, 
        iou: float = 0.5,
    ) -> Tuple[int, str]:
        """
        Predict and count using ensemble approach.
        
        Args:
            image: PIL Image
            target_class_name: Class name to count (will be normalized)
            conf: Confidence threshold
            iou: IoU threshold
            
        Returns:
            Tuple of (count, model_used) where model_used is 'primary', 'secondary', 'base', or 'none'
        """
        # Normalize class name
        normalized_class = normalize_class_name(target_class_name)
        LOGGER.info(f"Original class name: '{target_class_name}' -> Normalized: '{normalized_class}'")
        
        # Try primary model first
        LOGGER.info("Running primary YOLO model...")
        primary_results = self.primary_counter.predict(image, conf=conf, iou=iou)
        LOGGER.info(f"Primary model class names: {self.primary_counter.get_class_names()}")
        primary_count = self.primary_counter.count_class(primary_results, normalized_class)
        LOGGER.info(f"Primary model count: {primary_count}")
        
        if primary_count > 0:
            # Primary model found something, use it
            return primary_count, 'primary'
        
        # Primary model found nothing, try secondary model
        LOGGER.info("Primary model found 0, trying secondary YOLO model...")
        secondary_results = self.secondary_counter.predict(image, conf=conf, iou=iou)
        LOGGER.info(f"Secondary model class names: {self.secondary_counter.get_class_names()}")
        secondary_count = self.secondary_counter.count_class(secondary_results, normalized_class)
        LOGGER.info(f"Secondary model count: {secondary_count}")
        
        if secondary_count > 0:
            # Secondary model found something
            return secondary_count, 'secondary'
        
        # Both primary and secondary found nothing, try base model if available
        if self.base_counter is not None:
            LOGGER.info("Primary and secondary models found 0, trying base YOLO model...")
            base_results = self.base_counter.predict(image, conf=conf, iou=iou)
            LOGGER.info(f"Base model class names: {self.base_counter.get_class_names()}")
            base_count = self.base_counter.count_class(base_results, normalized_class)
            LOGGER.info(f"Base model count: {base_count}")
            
            if base_count > 0:
                # Base model found something
                return base_count, 'base'
        
        # All models found nothing
        LOGGER.warning(f"All models returned 0 for class '{normalized_class}' (original: '{target_class_name}')")
        return 0, 'none'
    
    def count_class(self, image: Image.Image, target_class_name: str, conf: float = 0.25, iou: float = 0.5) -> int:
        """
        Simplified interface that just returns count (uses ensemble logic internally).
        """
        count, _ = self.predict_and_count(image, target_class_name, conf, iou)
        return count


def build_numeric_question_with_yolo_context(
    original_question: str, 
    yolo_count: int,
    include_yolo_context: bool = True
) -> str:
    """
    Build the numeric question prompt with YOLO count as context.
    
    Args:
        original_question: Original counting question
        yolo_count: Count from YOLO ensemble
        include_yolo_context: Whether to include YOLO count as context
        
    Returns:
        Formatted question string with numeric instruction
    """
    # Use the specific numeric instruction format
    numeric_instruction = "Answer the following numeric question about the image. Provide only the number."
    
    if include_yolo_context:
        # Add YOLO count as context
        yolo_context = f"A detection model initially found {yolo_count} object(s). "
        question_text = f"{yolo_context}{original_question}"
    else:
        question_text = original_question
    
    # Combine instruction and question
    full_prompt = f"{numeric_instruction}\n{question_text}"
    
    return full_prompt


def load_yolo_models():
    """
    Load YOLO models for counting ensemble.
    Tries to load three models: yolo1, yolo2, and base (yolo11l-obb.pt).
    
    Returns:
        Tuple of (yolo_model_1, yolo_model_2, yolo_base) or (None, None, None) if not available
    """
    if not YOLO_AVAILABLE:
        LOGGER.warning("YOLO not available, counting ensemble will use Qwen only")
        return None, None, None
    
    yolo_model_1 = None
    yolo_model_2 = None
    yolo_base = None
    
    # Try to load primary YOLO model (same as used in grounding.py)
    try:
        try:
            yolo_model_1 = YOLO("./runs_obb_train4_weights_last.pt")
            LOGGER.info("✅ YOLO model 1 loaded from ./runs_obb_train4_weights_last.pt")
        except:
            yolo_model_1 = YOLO("yolo11l-obb.pt")
            LOGGER.info("✅ YOLO model 1 loaded from yolo11l-obb.pt")
    except Exception as e:
        LOGGER.warning(f"Failed to load YOLO model 1: {e}")
    
    # Try to load secondary YOLO model (if available)
    # You can add additional model paths here if you have multiple YOLO models
    try:
        try:
            yolo_model_2 = YOLO("./last.pt")
            LOGGER.info("✅ YOLO model 2 loaded from ./last.pt")
        except:
            yolo_model_2 = YOLO("yolo11l-obb.pt")
            LOGGER.info("✅ YOLO model 2 loaded from yolo11l-obb.pt")
    except Exception as e:
        LOGGER.warning(f"Failed to load YOLO model 2: {e}")
    
    # Try to load base YOLO model (yolo11l-obb.pt)
    try:
        yolo_base = YOLO("yolo11l-obb.pt")
        LOGGER.info("✅ Base YOLO model loaded from yolo11l-obb.pt")
    except Exception as e:
        LOGGER.warning(f"Failed to load base YOLO model: {e}")
    
    return yolo_model_1, yolo_model_2, yolo_base  

def ensemble_counting(
    image: Image.Image,
    image_url: str,
    question: str,
    object_class: Optional[str] = None,
    yolo_model_1=None,
    yolo_model_2=None,
    yolo_base=None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
) -> Dict:
    """
    Ensemble counting using YOLO + base Qwen model (via modal).
    
    Workflow:
    1. YOLO ensemble gives initial count (using object_class if provided)
    2. Base Qwen model refines the count using YOLO output as context
    3. Final prediction comes from base Qwen
    
    Args:
        image: PIL Image (for YOLO processing)
        image_url: Image URL (for modal request)
        question: Original counting question
        object_class: Object class name from classifier (optional)
        yolo_model_1: Primary YOLO model
        yolo_model_2: Secondary YOLO model
        yolo_base: Base YOLO model (yolo11l-obb.pt)
        conf_thres: YOLO confidence threshold
        iou_thres: YOLO IoU threshold
    
    Returns:
        Dictionary with:
            - 'count': Final count (int)
            - 'yolo_count': YOLO ensemble count (int)
            - 'qwen_output': Raw Qwen output (str)
            - 'method': Method used ('ensemble' or 'yolo_only')
    """
    # Step 1: Get YOLO ensemble count
    yolo_count = 0
    yolo_method = 'none'
    
    if object_class and yolo_model_1 is not None and yolo_model_2 is not None:
        try:
            ensemble_counter = EnsembleYOLOCounter(yolo_model_1, yolo_model_2, yolo_base)
            yolo_count, yolo_method = ensemble_counter.predict_and_count(
                image,
                object_class,
                conf=conf_thres,
                iou=iou_thres,
            )
            LOGGER.info(f"YOLO ensemble count: {yolo_count} (model: {yolo_method}) for class: {object_class}")
        except Exception as e:
            LOGGER.warning(f"YOLO ensemble counting failed: {e}, using Qwen only")
            yolo_count = 0
            yolo_method = 'error'
    else:
        if not object_class:
            LOGGER.warning("No object_class provided for counting, using Qwen only")
        else:
            LOGGER.warning("YOLO models not available, using Qwen only")
    
    # Step 2: Get final count from base Qwen model using YOLO count as context
    try:
        # Build question with YOLO context
        question_with_context = build_numeric_question_with_yolo_context(
            question,
            yolo_count,
            include_yolo_context=(yolo_count > 0 or yolo_method != 'none')
        )
        
        # Call base Qwen via modal with image URL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": question_with_context + " Answer in only a number."}
                ]
            }
        ]

        print(f"Messages: {messages}")
        
        qwen_output = qwen_base(messages)
        print(f"Qwen output: {qwen_output}")
        # qwen_output = "10"
        
        # Extract count from Qwen output
        final_count = extract_count(qwen_output)
        if final_count is None:
            # If parsing fails, fallback to YOLO count
            LOGGER.warning(f"Failed to parse Qwen output '{qwen_output}', falling back to YOLO count")
            final_count = yolo_count if yolo_count > 0 else 0
            method = 'yolo_only' if yolo_count > 0 else 'qwen_parse_failed'
        else:
            method = 'ensemble'
        
        LOGGER.info(f"Final count: {final_count} (method: {method}, YOLO: {yolo_count})")
        
        return {
            'count': final_count,
            'yolo_count': yolo_count,
            'qwen_output': qwen_output,
            'method': method,
            'yolo_method': yolo_method,
        }
        
    except Exception as e:
        LOGGER.warning(f"Base Qwen inference failed: {e}, using YOLO count only")
        return {
            'count': yolo_count if yolo_count > 0 else 0,
            'yolo_count': yolo_count,
            'qwen_output': f"ERROR: {str(e)}",
            'method': 'yolo_only' if yolo_count > 0 else 'error',
            'yolo_method': yolo_method,
        }


def calculate_mask_area(mask: np.ndarray) -> float:
    """Calculate area of mask in pixels."""
    return float(np.sum(mask > 0.5))


def calculate_mask_perimeter(mask: np.ndarray) -> float:
    """Calculate perimeter of mask using contour detection."""
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Sum perimeter of all contours
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    return float(perimeter)


def calculate_mask_max_length(mask: np.ndarray) -> float:
    """Calculate maximum length (longest distance) across mask."""
    mask_binary = (mask > 0.5).astype(np.uint8)
    y_coords, x_coords = np.where(mask_binary > 0)
    
    if len(x_coords) == 0:
        return 0.0
    
    # Get all points in the mask
    points = np.column_stack((x_coords, y_coords))
    
    if len(points) < 2:
        return 0.0
    
    # Calculate pairwise distances and return maximum
    distances = cdist(points, points, metric='euclidean')
    max_length = float(np.max(distances))
    return max_length


def _decode_mask(mask_data: str, image_shape: Optional[tuple] = None) -> np.ndarray:
    """
    Decode mask from base64 format.
    
    Args:
        mask_data: Base64-encoded mask image
        image_shape: Optional (H, W) for validation
    
    Returns:
        Binary mask as numpy array (0/1)
    """
    try:
        # Decode base64
        mask_bytes = base64.b64decode(mask_data)
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert('L')
        mask = np.array(mask_image) > 128  # Binarize at 128 threshold
        return mask.astype(np.uint8)
    except Exception as e:
        LOGGER.error(f"Failed to decode mask: {e}")
        raise


def _fallback_size_measurement(image_url: str, prompt: str) -> float:
    """Fallback to base Qwen if SAM ensemble is not available."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": f"Answer the following numeric question about the image. Provide only the number.\n{prompt}"}
            ]
        }
    ]
    
    try:
        response = qwen_base(messages)
        parsed_result = _parse_numeric_answer(response)
        if isinstance(parsed_result, float):
            return parsed_result
        else:
            LOGGER.warning(f"Could not extract number from response: {response}, parsed as: {parsed_result}")
            return 0.0
    except Exception as e:
        LOGGER.error(f"Failed to get response from base Qwen: {e}")
        raise


def size_measurement_vqa(
    image_url: str,
    prompt: str,
    parameter_type: str,
    object_class: Optional[str] = None,
    spatial_resolution_m: float = 1.0
) -> float:
    """
    Handle size measurement queries (area, perimeter, length) using SAM ensemble.
    
    Args:
        image_url: URL of the image to process
        prompt: User query/instruction
        parameter_type: Type of measurement - "area", "perimeter", or "length"
        object_class: Optional object class name from classification
        spatial_resolution_m: Spatial resolution in meters per pixel (default: 1.0)
    
    Returns:
        Float value representing the numeric answer in real-world units (meters or square meters)
    """
    LOGGER.info(f"Processing {parameter_type} measurement query with object_class: {object_class}")
    
    # Step 1: Get OBB box if object_class is provided
    obb_box = None
    if object_class:
        LOGGER.info(f"Getting OBB box for object_class: {object_class}")
        try:
            # Call grounding to get OBB box
            grounding_results = grounding(
                image_url=image_url,
                instruction=f"Find the {object_class}",
            )
            
            if grounding_results and len(grounding_results) > 0:
                obb_box = grounding_results[0].get('obbox', None)
                LOGGER.info(f"Got OBB box from grounding: {obb_box}")
            else:
                LOGGER.warning(f"No OBB box found from grounding for {object_class}")
        except Exception as e:
            LOGGER.warning(f"Failed to get OBB box from grounding: {e}")
    
    # Step 2: If no OBB box, fallback to base Qwen
    if obb_box is None:
        LOGGER.warning("No OBB box available, cannot use SAM ensemble. Falling back to base Qwen.")
        return _fallback_size_measurement(image_url, prompt)
    
    # Step 3: Call SAM ensemble via Modal
    try:
        # Use object_class as text query for RemoteCLIP scoring
        text_query = object_class if object_class else prompt
        
        LOGGER.info(f"Calling SAM ensemble with OBB: {obb_box}, text_query: {text_query}")
        mask_result = sam_ensemble_mask(image_url, obb_box, text_query)
        
        # Step 4: Extract mask from result
        if mask_result.get('status') != 'success' or 'mask' not in mask_result:
            LOGGER.warning(f"SAM ensemble failed: {mask_result.get('message', 'Unknown error')}, falling back to base Qwen")
            return _fallback_size_measurement(image_url, prompt)
        
        mask_data = mask_result['mask']
        mask = _decode_mask(mask_data)
        LOGGER.info(f"Decoded mask with shape: {mask.shape}")
        
        # Step 5: Calculate measurement based on parameter_type
        LOGGER.info(f"Using spatial resolution: {spatial_resolution_m} m/pixel")
        
        if parameter_type == "area":
            measurement_pixels = calculate_mask_area(mask)
            # Convert area: pixels * (m/pixel)^2 = m^2
            measurement = measurement_pixels * (spatial_resolution_m ** 2)
            LOGGER.info(f"Calculated area: {measurement_pixels} pixels -> {measurement} m² (spatial_resolution_m={spatial_resolution_m})")
        elif parameter_type == "perimeter":
            measurement_pixels = calculate_mask_perimeter(mask)
            # Convert perimeter: pixels * m/pixel = m
            measurement = measurement_pixels * spatial_resolution_m
            LOGGER.info(f"Calculated perimeter: {measurement_pixels} pixels -> {measurement} m (spatial_resolution_m={spatial_resolution_m})")
        elif parameter_type == "length":
            measurement_pixels = calculate_mask_max_length(mask)
            # Convert length: pixels * m/pixel = m
            measurement = measurement_pixels * spatial_resolution_m
            LOGGER.info(f"Calculated max length: {measurement_pixels} pixels -> {measurement} m (spatial_resolution_m={spatial_resolution_m})")
        else:
            LOGGER.warning(f"Unknown parameter_type: {parameter_type}, falling back to base Qwen")
            return _fallback_size_measurement(image_url, prompt)
        
        return float(measurement)
        
    except Exception as e:
        LOGGER.error(f"SAM ensemble failed: {e}, falling back to base Qwen")
        return _fallback_size_measurement(image_url, prompt)


def numeric_vqa(
    image_url: str, 
    prompt: str,
    classification=None,
    yolo_model_1=None,
    yolo_model_2=None,
    yolo_base=None,
    spatial_resolution_m: float = 1.0
) -> float:
    """
    Main numeric VQA function that handles counting, size measurements, and other numeric queries.
    
    Args:
        image_url: URL of the image to process
        prompt: User query/instruction
        classification: Optional QueryClassification object with vqa_float_subtype and object_class
        yolo_model_1: Optional primary YOLO model (if None, will try to load)
        yolo_model_2: Optional secondary YOLO model (if None, will try to load)
        yolo_base: Optional base YOLO model (if None, will try to load)
        spatial_resolution_m: Spatial resolution in meters per pixel (default: 1.0)
    
    Returns:
        Float value representing the numeric answer
    """
    # Extract classification if not provided
    if classification is None:
        LOGGER.info("Classification not provided, extracting from prompt")
        try:
            classification = classify_query(prompt)
            if classification.task_type != "vqa_float":
                raise ValueError(f"Expected vqa_float task type, got {classification.task_type}")
        except Exception as e:
            LOGGER.error(f"Failed to extract classification: {e}")
            raise

    # Validate classification
    if classification.task_type != "vqa_float":
        raise ValueError(f"Expected vqa_float task type, got {classification.task_type}")

    # Extract subtype and object_class
    print(classification)
    vqa_float_subtype = getattr(classification, 'vqa_float_subtype', None)
    object_class = getattr(classification, 'object_class', None)

    LOGGER.info(f"Processing vqa_float query with subtype: {vqa_float_subtype}, object_class: {object_class}")

    # Load image
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        LOGGER.error(f"Failed to load image from URL: {e}")
        raise

    # Route to appropriate pipeline based on subtype
    if vqa_float_subtype == "counting":
        # Pipeline 1: Counting (YOLO ensemble + base Qwen)
        LOGGER.info("Routing to counting pipeline")

        # Load YOLO models if needed and not provided
        if object_class and (yolo_model_1 is None or yolo_model_2 is None or yolo_base is None):
            yolo_model_1, yolo_model_2, yolo_base = load_yolo_models()

        # Use counting ensemble for counting queries with object_class
        if object_class and yolo_model_1 is not None and yolo_model_2 is not None:
            LOGGER.info("Using counting ensemble (YOLO + base Qwen) for counting query")

            # Run ensemble counting (this already calls Qwen internally)
            ensemble_result = ensemble_counting(
                image=image,
                image_url=image_url,
                question=prompt,
                object_class=object_class,
                yolo_model_1=yolo_model_1,
                yolo_model_2=yolo_model_2,
                yolo_base=yolo_base,
                conf_thres=0.25,
                iou_thres=0.5,
            )

            # Get the final count from ensemble result
            final_count = ensemble_result['count']
            return float(final_count)
        else:
            # Counting without YOLO (fallback to base Qwen)
            LOGGER.info("Counting query detected but YOLO models not available or no object_class, using base model only")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url},
                        {"type": "text", "text": f"Answer the following numeric question about the image. Provide only the number.\n{prompt}"}
                    ]
                }
            ]
            try:
                response = qwen_base(messages)
                parsed_result = _parse_numeric_answer(response)
                if isinstance(parsed_result, float):
                    return parsed_result
                else:
                    LOGGER.warning(f"Could not extract number from response: {response}, parsed as: {parsed_result}")
                    return 0.0
            except Exception as e:
                LOGGER.error(f"Failed to get response from base Qwen: {e}")
                raise

    elif vqa_float_subtype in ["area", "perimeter", "length"]:
        # Pipeline 2: Size measurements (area, perimeter, length)
        LOGGER.info(f"Routing to size measurement pipeline for subtype: {vqa_float_subtype}")
        return size_measurement_vqa(image_url, prompt, vqa_float_subtype, object_class, spatial_resolution_m)

    else:
        # Pipeline 3: Other numeric queries
        LOGGER.info("Routing to other numeric query pipeline")

        # Call base Qwen via modal
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": f"Answer the following numeric question about the image. Provide only the number.\n{prompt}"}
                ]
            }
        ]

        try:
            response = qwen_base(messages)
            # Extract numeric value from response using same parsing as tasks.py
            parsed_result = _parse_numeric_answer(response)
            if isinstance(parsed_result, float):
                return parsed_result
            else:
                LOGGER.warning(f"Could not extract number from response: {response}, parsed as: {parsed_result}")
                return 0.0
        except Exception as e:
            LOGGER.error(f"Failed to get response from base Qwen: {e}")
            raise
