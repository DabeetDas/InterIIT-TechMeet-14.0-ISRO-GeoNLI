"""
Grounding module with YOLO ensemble support.

This module handles grounding queries using:
- Qwen fine-tuned grounding model (via modal)
- YOLO ensemble (3 models: primary, secondary, base) for object detection
- Ensemble logic to combine Qwen and YOLO predictions
"""

import logging
import math
import re
import requests
import io
from typing import Dict, List, Optional, Tuple, Sequence
from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from modal_request import qwen_ft_ground

LOGGER = logging.getLogger(__name__)

# Ensemble logic parameter
PROPORTION_AREA = 0.3

# Class name normalization mapping (similar to numeric_vqa.py)
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
    "vehicle": "Small Vehicle",
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
    
    # Return original if no match found
    LOGGER.warning(f"Could not normalize class name: {class_name}, using as-is")
    return class_name


def _convert_angle_to_cv2_minarearect_format(angle_deg: float) -> float:
    """
    Convert angle to cv2.minAreaRect() format ([-90, 0) degrees).
    
    Args:
        angle_deg: Angle in degrees (any range)
    
    Returns:
        Angle in cv2.minAreaRect() format ([-90, 0) degrees)
    """
    # Normalize to [0, 180)
    angle_deg = angle_deg % 180
    
    # Convert to [-90, 0) range
    if angle_deg >= 90:
        angle_deg = angle_deg - 180
    else:
        angle_deg = angle_deg - 90
    
    return angle_deg


def _normalized_to_pct(obb_normalized: Sequence[float]) -> Tuple[float, float, float, float, float]:
    """
    Convert normalized OBB (0-1) to percentage format (0-100).
    
    Args:
        obb_normalized: [cx, cy, w, h, angle] in normalized format (0-1)
    
    Returns:
        Tuple of (cx_pct, cy_pct, w_pct, h_pct, angle_deg) in percentage format
    """
    cx, cy, w, h, angle = obb_normalized
    return (cx * 100.0, cy * 100.0, w * 100.0, h * 100.0, angle)


def _pct_to_normalized(obb_pct: Sequence[float]) -> Tuple[float, float, float, float, float]:
    """
    Convert percentage OBB (0-100) to normalized format (0-1).
    
    Args:
        obb_pct: [cx_pct, cy_pct, w_pct, h_pct, angle_deg] in percentage format
    
    Returns:
        Tuple of (cx, cy, w, h, angle) in normalized format (0-1)
    """
    cx_pct, cy_pct, w_pct, h_pct, angle = obb_pct
    return (cx_pct / 100.0, cy_pct / 100.0, w_pct / 100.0, h_pct / 100.0, angle)


def _xywhr_to_normalized(box_xywhr: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float, float]:
    """
    Convert YOLO xywhr format to normalized format (0-1).
    
    Args:
        box_xywhr: [cx, cy, w, h, theta] in pixel coordinates
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Tuple of (cx, cy, w, h, theta_deg) in normalized format (0-1)
    """
    # Make sure we have a proper list/array and extract values
    if isinstance(box_xywhr, np.ndarray):
        box_xywhr = box_xywhr.tolist()
    
    cx, cy, w, h, theta = box_xywhr
    theta_deg = math.degrees(theta)
    
    # Normalize angle to [-90, 0) for cv2.minAreaRect() format
    # Note: The swap logic should happen after understanding the angle format
    # For now, let's normalize the angle first, then decide if swap is needed
    theta_deg_normalized = _convert_angle_to_cv2_minarearect_format(theta_deg)
    
    # The swap logic might be incorrect - removing it for now
    # If YOLO outputs are already in the correct format, we shouldn't swap
    # If swap is needed, it should be based on the normalized angle, not the original
    return (
        cx / width,
        cy / height,
        w / width,
        h / height,
        theta_deg_normalized,
    )


def _box_to_polygon_pixels(obb_normalized: Sequence[float], img_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert normalized OBB to polygon in pixel coordinates.
    
    Args:
        obb_normalized: [cx, cy, w, h, angle] in normalized format (0-1)
        img_size: (width, height) of image in pixels
    
    Returns:
        Array of 4 corner coordinates in pixel space
    """
    cx_norm, cy_norm, w_norm, h_norm, theta_deg = obb_normalized
    W, H = img_size
    cx = cx_norm * W
    cy = cy_norm * H
    w = w_norm * W
    h = h_norm * H
    th = math.radians(theta_deg)
    dx = w / 2.0
    dy = h / 2.0
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=float)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=float)
    pts = (corners @ R.T) + np.array([cx, cy])
    return pts


def calculate_obb_iou(box_a: Sequence[float], box_b: Sequence[float], img_size: Tuple[int, int]) -> float:
    """
    Calculate rotated IoU between two OBBs in normalized format.
    
    Args:
        box_a: [cx, cy, w, h, angle] in normalized format (0-1)
        box_b: [cx, cy, w, h, angle] in normalized format (0-1)
        img_size: (width, height) of image in pixels
    
    Returns:
        IoU value between 0 and 1
    """
    pts_a = _box_to_polygon_pixels(box_a, img_size)
    pts_b = _box_to_polygon_pixels(box_b, img_size)
    
    if SHAPELY_AVAILABLE:
        try:
            poly_a = ShapelyPolygon(pts_a)
            poly_b = ShapelyPolygon(pts_b)
            if not poly_a.is_valid or not poly_b.is_valid:
                return 0.0
            inter = poly_a.intersection(poly_b).area
            union = poly_a.union(poly_b).area
            if union <= 0:
                return 0.0
            return float(inter / union)
        except Exception:
            pass
    
    # Fallback to axis-aligned IoU
    min_ax, min_ay = pts_a.min(axis=0)
    max_ax, max_ay = pts_a.max(axis=0)
    min_bx, min_by = pts_b.min(axis=0)
    max_bx, max_by = pts_b.max(axis=0)
    ix_min = max(min_ax, min_bx)
    iy_min = max(min_ay, min_by)
    ix_max = min(max_ax, max_bx)
    iy_max = min(max_ay, max_by)
    iw = max(0.0, ix_max - ix_min)
    ih = max(0.0, iy_max - iy_min)
    inter = iw * ih
    area_a = max(0.0, (max_ax - min_ax)) * max(0.0, (max_ay - min_ay))
    area_b = max(0.0, (max_bx - min_bx)) * max(0.0, (max_by - min_by))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def parse_qwen_response(qwen_response: Dict) -> List[Dict]:
    """
    Parse Qwen grounding response JSON.
    
    Args:
        qwen_response: Response from qwen_ft_ground in format:
            {'response': [{'object-id': '1', 'obbox': [cx, cy, w, h, angle]}, ...]}
    
    Returns:
        List of OBB dictionaries with normalized format
    """
    results = []
    if not isinstance(qwen_response, dict):
        LOGGER.warning(f"Invalid qwen_response type: {type(qwen_response)}")
        return results
    
    response_list = qwen_response.get('response', [])
    if not isinstance(response_list, list):
        LOGGER.warning(f"Invalid response format: {response_list}")
        return results
    
    for entry in response_list:
        if not isinstance(entry, dict):
            continue
        obbox = entry.get('obbox', [])
        if not isinstance(obbox, (list, tuple)) or len(obbox) != 5:
            continue
        
        # Ensure values are floats and in valid range
        try:
            cx, cy, w, h, angle = map(float, obbox)
            # Normalize angle to cv2.minAreaRect() format
            angle = _convert_angle_to_cv2_minarearect_format(angle)
            results.append({
                'object-id': entry.get('object-id', str(len(results) + 1)),
                'obbox': [cx, cy, w, h, angle]
            })
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Error parsing obbox {obbox}: {e}")
            continue
    
    return results


def run_yolo_obb(
    image: Image.Image,
    model,
    conf_thres: float = 0.2,
    iou_thres: float = 0.45,
    max_det: int = 200,
    imgsz: int = 640,
) -> List[Dict]:
    """
    Run YOLO OBB detector on image and return results in normalized format.
    
    Args:
        image: PIL Image
        model: YOLO model
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections
        imgsz: Image size for inference
    
    Returns:
        List of detection dictionaries with normalized obbox format
    """
    if not YOLO_AVAILABLE or model is None:
        LOGGER.warning("YOLO not available or model is None, skipping YOLO detection")
        return []
    
    try:
        results = model.predict(
            source=image,
            save=False,
            show=False,
            conf=float(conf_thres),
            iou=float(iou_thres),
            max_det=int(max_det),
            # imgsz=int(imgsz),
            verbose=False,
        )
        
        detections = []
        first = results[0]
        W, H = image.size

        print(f"First: {first.obb}")
        
        if getattr(first, "obb", None) is not None and getattr(first.obb, "xywhr", None) is not None:
            boxes = first.obb.xywhr.cpu().numpy()  # shape (N,5)
            confs = first.obb.conf.cpu().numpy()
            cls_ids = first.obb.cls.cpu().numpy()  # shape (N,)
            
            # Debug: Print box shapes and first few boxes
            LOGGER.info(f"Boxes shape: {boxes.shape}, Number of detections: {len(boxes)}")
            if len(boxes) > 0:
                LOGGER.info(f"First box (raw): {boxes[0]}")
                if len(boxes) > 1:
                    LOGGER.info(f"Second box (raw): {boxes[1]}")
            
            # Get class names map
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                names_map = model.model.names
            elif hasattr(model, 'names'):
                names_map = model.names
            else:
                names_map = {}
            
            # CRITICAL FIX: Explicitly iterate over rows and copy each box
            for i in range(len(boxes)):
                b = boxes[i].copy()  # Make a copy to avoid view issues
                c = float(confs[i])
                cls_id = int(cls_ids[i])
                cls_name = names_map.get(cls_id, str(cls_id))
                
                obb_normalized = _xywhr_to_normalized(b, W, H)
                
                # Debug: Print converted box
                LOGGER.info(f"Box {i}: raw={b}, normalized={obb_normalized}")
                
                detections.append({
                    "obbox": list(obb_normalized),
                    "conf": c,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                })
        
        return detections
    except Exception as e:
        LOGGER.warning(f"Error running YOLO model: {e}")
        return []


def ensemble_single_object_grounding(
    qwen_obbs: List[Dict],
    image: Image.Image,
    yolo_model_1,
    yolo_model_2,
    yolo_base=None,
    object_class: Optional[str] = None,
) -> List[Dict]:
    """
    Ensemble grounding for single object using Qwen + three YOLO models.
    
    Args:
        qwen_obbs: List of OBB dictionaries from Qwen model
        image: PIL Image
        yolo_model_1: Primary YOLO model
        yolo_model_2: Secondary YOLO model
        yolo_base: Base YOLO model (fallback)
        object_class: Object class name from classifier (optional)
    
    Returns:
        List of OBB dictionaries with ensemble results
    """
    W, H = image.size
    
    # Get first Qwen prediction (or default)
    qwen_obb = None
    if qwen_obbs and len(qwen_obbs) > 0:
        qwen_obb = qwen_obbs[0].get('obbox', [0.5, 0.5, 0.1, 0.1, 0.0])
    else:
        qwen_obb = [0.5, 0.5, 0.1, 0.1, 0.0]
    
    # Run YOLO models and collect ALL detections from all models
    yolo_boxes = []
    yolo_confs = []
    yolo_cls_names = []
    
    # Helper to process YOLO results - collect all detections
    def collect_all_results(model, model_name=""):
        detections = run_yolo_obb(image, model)
        for det in detections:
            yolo_boxes.append(det["obbox"])
            yolo_confs.append(det["conf"])
            yolo_cls_names.append(det["cls_name"])
        LOGGER.info(f"Collected {len(detections)} detections from {model_name}")
    
    # Run all three YOLO models and collect ALL detections
    if yolo_model_1 is not None:
        collect_all_results(yolo_model_1, "Model 1 (Primary)")
    
    if yolo_model_2 is not None:
        collect_all_results(yolo_model_2, "Model 2 (Secondary)")
    
    if yolo_base is not None:
        collect_all_results(yolo_base, "Base Model")
    
    # Print final collected YOLO OBBs
    print(f"Final collected YOLO OBBs (single object): {len(yolo_boxes)} detections")
    for idx, (ybox, conf, cls_name) in enumerate(zip(yolo_boxes, yolo_confs, yolo_cls_names)):
        print(f"  YOLO OBB {idx+1}: obbox={ybox}, conf={conf:.3f}, class={cls_name}")
    
    # Ensemble Logic
    ensemble_obb = qwen_obb  # Default to Qwen
    chosen_conf = 0.0
    best_idx = -1
    method = "Qwen (Fallback)"
    
    if len(yolo_boxes) == 0:
        LOGGER.info("No YOLO candidates found — returning Qwen box.")
    else:
        # 1. Compute IoU
        yolo_ious = []
        max_iou = -1.0
        max_iou_idx = -1
        
        for i, ybox in enumerate(yolo_boxes):
            iou = calculate_obb_iou(qwen_obb, ybox, (W, H))
            yolo_ious.append(iou)
            if iou > max_iou:
                max_iou = iou
                max_iou_idx = i
        
        if max_iou > 0:
            best_idx = max_iou_idx
            ensemble_obb = yolo_boxes[best_idx]
            chosen_conf = yolo_confs[best_idx]
            method = "Max IoU"
        else:
            # 2. IoU is zero -> Class Matching + Spatial Search
            LOGGER.info("IoU is zero. Performing class matching and spatial search...")
            
            if object_class:
                # Normalize object class name
                normalized_class = normalize_class_name(object_class)
                
                # Class Matching + Spatial Search
                q_cx, q_cy = qwen_obb[0], qwen_obb[1]
                q_cx_px = q_cx * W
                q_cy_px = q_cy * H
                
                # Square side length = PROPORTION_AREA * image height
                side_len = PROPORTION_AREA * H
                half_side = side_len / 2.0
                
                # Filter YOLO boxes by class name
                candidate_indices = []
                for i, name in enumerate(yolo_cls_names):
                    normalized_name = normalize_class_name(name)
                    if name == normalized_class or normalized_name == normalized_class:
                        candidate_indices.append(i)
                
                nearest_dist = float('inf')
                nearest_idx = -1
                
                for idx in candidate_indices:
                    ybox = yolo_boxes[idx]
                    y_cx, y_cy = ybox[0], ybox[1]
                    y_cx_px = y_cx * W
                    y_cy_px = y_cy * H
                    
                    # Check if inside square
                    dx = abs(y_cx_px - q_cx_px)
                    dy = abs(y_cy_px - q_cy_px)
                    
                    if dx <= half_side and dy <= half_side:
                        dist = math.hypot(dx, dy)  # Euclidean distance to center
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_idx = idx
                
                if nearest_idx != -1:
                    best_idx = nearest_idx
                    ensemble_obb = yolo_boxes[best_idx]
                    chosen_conf = yolo_confs[best_idx]
                    method = f"Spatial Search (class: {object_class})"
                else:
                    LOGGER.info(f"No matching YOLO box found in spatial region for class: {object_class}")
            else:
                LOGGER.info("No object_class provided, cannot perform class matching")
    
    LOGGER.info(f"Ensemble method: {method}, confidence: {chosen_conf:.3f}")
    
    return [{
        "object-id": "1",
        "obbox": ensemble_obb
    }]


def run_yolo_ensemble_multi(
    image: Image.Image,
    yolo_model_1,
    yolo_model_2,
    yolo_base=None,
    object_class: Optional[str] = None,
    conf_thres: float = 0.2,
    iou_thres: float = 0.45,
    image_url: Optional[str] = None,
    instruction: Optional[str] = None,
) -> List[Dict]:
    """
    Run YOLO ensemble for multiple objects with Qwen fallback.
    
    Args:
        image: PIL Image
        yolo_model_1: Primary YOLO model
        yolo_model_2: Secondary YOLO model
        yolo_base: Base YOLO model (fallback)
        object_class: Object class name from classifier (optional)
        conf_thres: Confidence threshold for YOLO
        iou_thres: IoU threshold for NMS
        image_url: Image URL for Qwen fallback (optional)
        instruction: Instruction text for Qwen fallback (optional)
    
    Returns:
        List of OBB dictionaries with ensemble results
    """
    # Check if any YOLO models are available
    yolo_models_available = (
        yolo_model_1 is not None or 
        yolo_model_2 is not None or 
        yolo_base is not None
    )
    
    # If no YOLO models available, use Qwen fallback
    if not yolo_models_available:
        LOGGER.warning("No YOLO models available, falling back to Qwen")
        if image_url is None or instruction is None:
            LOGGER.error("Cannot use Qwen fallback: image_url and instruction are required")
            return []
        return _qwen_fallback_multi(image_url, instruction)
    
    # If object_class is provided, use YOLO ensemble to get detections
    if object_class:
        normalized_class = normalize_class_name(object_class)
        
        # Try primary model first
        primary_detections = []
        if yolo_model_1 is not None:
            primary_detections = run_yolo_obb(image, yolo_model_1, conf_thres, iou_thres)
            print(f"Primary detections shape: {len(primary_detections)}")
            primary_detections = [
                det for det in primary_detections
                if det["cls_name"] == normalized_class or normalize_class_name(det["cls_name"]) == normalized_class
            ]
        
        if len(primary_detections) > 0:
            LOGGER.info(f"Primary YOLO model found {len(primary_detections)} detections for class '{normalized_class}'")
            # Print final collected YOLO OBBs
            print(f"Final collected YOLO OBBs (multi-object, primary model): {len(primary_detections)} detections")
            for idx, det in enumerate(primary_detections):
                print(f"  YOLO OBB {idx+1}: obbox={det['obbox']}, conf={det['conf']:.3f}, class={det['cls_name']}")
            # Sort by confidence and return
            primary_detections.sort(key=lambda x: x['conf'], reverse=True)
            results = []
            for idx, det in enumerate(primary_detections):
                results.append({
                    "object-id": str(idx + 1),
                    "obbox": det["obbox"]
                })
            return results
        
        # Primary model found nothing, try secondary model
        secondary_detections = []
        if yolo_model_2 is not None:
            secondary_detections = run_yolo_obb(image, yolo_model_2, conf_thres, iou_thres)
            secondary_detections = [
                det for det in secondary_detections
                if det["cls_name"] == normalized_class or normalize_class_name(det["cls_name"]) == normalized_class
            ]
        
        if len(secondary_detections) > 0:
            LOGGER.info(f"Secondary YOLO model found {len(secondary_detections)} detections for class '{normalized_class}'")
            # Print final collected YOLO OBBs
            print(f"Final collected YOLO OBBs (multi-object, secondary model): {len(secondary_detections)} detections")
            for idx, det in enumerate(secondary_detections):
                print(f"  YOLO OBB {idx+1}: obbox={det['obbox']}, conf={det['conf']:.3f}, class={det['cls_name']}")
            # Sort by confidence and return
            secondary_detections.sort(key=lambda x: x['conf'], reverse=True)
            results = []
            for idx, det in enumerate(secondary_detections):
                results.append({
                    "object-id": str(idx + 1),
                    "obbox": det["obbox"]
                })
            return results
        
        # Both primary and secondary found nothing, try base model
        if yolo_base is not None:
            base_detections = run_yolo_obb(image, yolo_base, conf_thres, iou_thres)
            base_detections = [
                det for det in base_detections
                if det["cls_name"] == normalized_class or normalize_class_name(det["cls_name"]) == normalized_class
            ]
            
            if len(base_detections) > 0:
                LOGGER.info(f"Base YOLO model found {len(base_detections)} detections for class '{normalized_class}'")
                # Print final collected YOLO OBBs
                print(f"Final collected YOLO OBBs (multi-object, base model): {len(base_detections)} detections")
                for idx, det in enumerate(base_detections):
                    print(f"  YOLO OBB {idx+1}: obbox={det['obbox']}, conf={det['conf']:.3f}, class={det['cls_name']}")
                # Sort by confidence and return
                base_detections.sort(key=lambda x: x['conf'], reverse=True)
                results = []
                for idx, det in enumerate(base_detections):
                    results.append({
                        "object-id": str(idx + 1),
                        "obbox": det["obbox"]
                    })
                return results
        
        # Zero detections from all YOLO models - use Qwen fallback
        LOGGER.warning(f"No YOLO detections found for class '{normalized_class}', falling back to Qwen")
        if image_url is not None and instruction is not None:
            return _qwen_fallback_multi(image_url, instruction)
        else:
            LOGGER.error("Cannot use Qwen fallback: image_url and instruction are required")
            return []
    
    # If no object_class, try Qwen fallback if possible
    if image_url is not None and instruction is not None:
        LOGGER.warning("No object_class provided, falling back to Qwen")
        return _qwen_fallback_multi(image_url, instruction)
    
    LOGGER.warning("No object_class provided and no Qwen fallback available")
    return []


def _qwen_fallback_multi(image_url: str, instruction: str) -> List[Dict]:
    """
    Qwen fallback for multi-object grounding.
    
    Args:
        image_url: URL of the image to process
        instruction: Grounding instruction/query
    
    Returns:
        List of OBB dictionaries from Qwen
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": "You are a grounding and OBB (Oriented Bounding Box) detection model. Locate the object described. [refer] "+instruction}
                ]
            }
        ]
        
        qwen_response = qwen_ft_ground(messages)
        LOGGER.info(f"Qwen grounding response (fallback): {qwen_response}")
        
        # Parse Qwen response
        qwen_obbs = parse_qwen_response(qwen_response)
        LOGGER.info(f"Parsed {len(qwen_obbs)} OBBs from Qwen response (fallback)")
        
        # Convert to results format
        results = []
        for idx, qwen_obb_dict in enumerate(qwen_obbs):
            results.append({
                "object-id": str(idx + 1),
                "obbox": qwen_obb_dict.get('obbox', [0.5, 0.5, 0.1, 0.1, 0.0])
            })
        
        return results
        
    except Exception as e:
        LOGGER.error(f"Failed to get Qwen grounding response (fallback): {e}")
        return []


def ensemble_multi_object_grounding(
    qwen_obbs: List[Dict],
    image: Image.Image,
    yolo_model_1,
    yolo_model_2,
    yolo_base=None,
    object_class: Optional[str] = None,
    conf_thres: float = 0.2,
    iou_thres: float = 0.45,
) -> List[Dict]:
    """
    Ensemble grounding for multiple objects using Qwen + three YOLO models.
    
    Args:
        qwen_obbs: List of OBB dictionaries from Qwen model
        image: PIL Image
        yolo_model_1: Primary YOLO model
        yolo_model_2: Secondary YOLO model
        yolo_base: Base YOLO model (fallback)
        object_class: Object class name from classifier (optional)
        conf_thres: Confidence threshold for YOLO
        iou_thres: IoU threshold for NMS
    
    Returns:
        List of OBB dictionaries with ensemble results
    """
    # If object_class is provided, use YOLO ensemble to get detections
    if object_class:
        normalized_class = normalize_class_name(object_class)
        
        # Try primary model first
        primary_detections = []
        if yolo_model_1 is not None:
            primary_detections = run_yolo_obb(image, yolo_model_1, conf_thres, iou_thres)
            primary_detections = [
                det for det in primary_detections
                if det["cls_name"] == normalized_class or normalize_class_name(det["cls_name"]) == normalized_class
            ]
        
        if len(primary_detections) > 0:
            LOGGER.info(f"Primary YOLO model found {len(primary_detections)} detections for class '{normalized_class}'")
            # Print final collected YOLO OBBs
            print(f"Final collected YOLO OBBs (multi-object, primary model): {len(primary_detections)} detections")
            for idx, det in enumerate(primary_detections):
                print(f"  YOLO OBB {idx+1}: obbox={det['obbox']}, conf={det['conf']:.3f}, class={det['cls_name']}")
            # Sort by confidence and return
            primary_detections.sort(key=lambda x: x['conf'], reverse=True)
            results = []
            for idx, det in enumerate(primary_detections):
                results.append({
                    "object-id": str(idx + 1),
                    "obbox": det["obbox"]
                })
            return results
        
        # Primary model found nothing, try secondary model
        secondary_detections = []
        if yolo_model_2 is not None:
            secondary_detections = run_yolo_obb(image, yolo_model_2, conf_thres, iou_thres)
            secondary_detections = [
                det for det in secondary_detections
                if det["cls_name"] == normalized_class or normalize_class_name(det["cls_name"]) == normalized_class
            ]
        
        if len(secondary_detections) > 0:
            LOGGER.info(f"Secondary YOLO model found {len(secondary_detections)} detections for class '{normalized_class}'")
            # Print final collected YOLO OBBs
            print(f"Final collected YOLO OBBs (multi-object, secondary model): {len(secondary_detections)} detections")
            for idx, det in enumerate(secondary_detections):
                print(f"  YOLO OBB {idx+1}: obbox={det['obbox']}, conf={det['conf']:.3f}, class={det['cls_name']}")
            # Sort by confidence and return
            secondary_detections.sort(key=lambda x: x['conf'], reverse=True)
            results = []
            for idx, det in enumerate(secondary_detections):
                results.append({
                    "object-id": str(idx + 1),
                    "obbox": det["obbox"]
                })
            return results
        
        # Both primary and secondary found nothing, try base model
        if yolo_base is not None:
            base_detections = run_yolo_obb(image, yolo_base, conf_thres, iou_thres)
            base_detections = [
                det for det in base_detections
                if det["cls_name"] == normalized_class or normalize_class_name(det["cls_name"]) == normalized_class
            ]
            
            if len(base_detections) > 0:
                LOGGER.info(f"Base YOLO model found {len(base_detections)} detections for class '{normalized_class}'")
                # Print final collected YOLO OBBs
                print(f"Final collected YOLO OBBs (multi-object, base model): {len(base_detections)} detections")
                for idx, det in enumerate(base_detections):
                    print(f"  YOLO OBB {idx+1}: obbox={det['obbox']}, conf={det['conf']:.3f}, class={det['cls_name']}")
                # Sort by confidence and return
                base_detections.sort(key=lambda x: x['conf'], reverse=True)
                results = []
                for idx, det in enumerate(base_detections):
                    results.append({
                        "object-id": str(idx + 1),
                        "obbox": det["obbox"]
                    })
                return results
        
        LOGGER.warning(f"No YOLO detections found for class '{normalized_class}', using Qwen predictions")
    
    # Fallback to Qwen predictions if no YOLO detections or no object_class
    results = []
    for idx, qwen_obb_dict in enumerate(qwen_obbs):
        obbox = qwen_obb_dict.get('obbox', [0.5, 0.5, 0.1, 0.1, 0.0])
        results.append({
            "object-id": str(idx + 1),
            "obbox": obbox
        })
    
    return results


def load_yolo_models():
    """
    Load YOLO models for grounding ensemble.
    Tries to load three models: yolo1, yolo2, and base (yolo11l-obb.pt).
    
    Returns:
        Tuple of (yolo_model_1, yolo_model_2, yolo_base) or (None, None, None) if not available
    """
    if not YOLO_AVAILABLE:
        LOGGER.warning("YOLO not available, grounding will use Qwen only")
        return None, None, None
    
    yolo_model_1 = None
    yolo_model_2 = None
    yolo_base = None
    
    # Try to load primary YOLO model
    try:
        try:
            yolo_model_1 = YOLO("./runs_obb_train4_weights_last.pt")
            LOGGER.info("✅ YOLO model 1 loaded from ./runs_obb_train4_weights_last.pt")
        except:
            yolo_model_1 = YOLO("yolo11l-obb.pt")
            LOGGER.info("✅ YOLO model 1 loaded from yolo11l-obb.pt")
    except Exception as e:
        LOGGER.warning(f"Failed to load YOLO model 1: {e}")
    
    # Try to load secondary YOLO model
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
        yolo_base = YOLO("yolo11x-obb.pt")
        LOGGER.info("✅ Base YOLO model loaded from yolo11x-obb.pt")
    except Exception as e:
        LOGGER.warning(f"Failed to load base YOLO model: {e}")
    
    return yolo_model_1, yolo_model_2, yolo_base


def grounding(
    image_url: str,
    instruction: str,
    yolo_model_1=None,
    yolo_model_2=None,
    yolo_base=None,
    object_class: Optional[str] = None,
    classification=None,
) -> List[Dict]:
    """
    Main grounding function that handles grounding queries using Qwen + YOLO ensemble.
    
    Args:
        image_url: URL of the image to process
        instruction: Grounding instruction/query
        yolo_model_1: Optional primary YOLO model (if None, will try to load)
        yolo_model_2: Optional secondary YOLO model (if None, will try to load)
        yolo_base: Optional base YOLO model (if None, will try to load)
        object_class: Optional object class name from classifier
        classification: Optional QueryClassification object (if provided, will extract object_class and subtype)
    
    Returns:
        List of OBB dictionaries in format:
            [{"object-id": "1", "obbox": [cx, cy, w, h, angle]}, ...]
        where obbox values are normalized (0-1) and angle is in cv2.minAreaRect() format
    """
    # Extract object_class and subtype from classification if provided

    if classification is not None:
        if object_class is None:
            object_class = getattr(classification, 'object_class', None)
        subtype = getattr(classification, 'grounding_subtype', None)
    else:
        subtype = None
    
    # Load image
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        LOGGER.error(f"Failed to load image from URL: {e}")
        raise
    
    # Load YOLO models if needed and not provided
    if yolo_model_1 is None or yolo_model_2 is None or yolo_base is None:
        loaded_yolo_1, loaded_yolo_2, loaded_yolo_base = load_yolo_models()
        if yolo_model_1 is None:
            yolo_model_1 = loaded_yolo_1
        if yolo_model_2 is None:
            yolo_model_2 = loaded_yolo_2
        if yolo_base is None:
            yolo_base = loaded_yolo_base
    
    # Check grounding_subtype first to determine pipeline
    if subtype == "multiple":
        # For multiple: skip Qwen, call YOLO ensemble directly (with Qwen fallback if needed)
        LOGGER.info("Grounding subtype is 'multiple': using YOLO ensemble (with Qwen fallback if needed)")
        return run_yolo_ensemble_multi(
            image=image,
            yolo_model_1=yolo_model_1,
            yolo_model_2=yolo_model_2,
            yolo_base=yolo_base,
            object_class=object_class,
            image_url=image_url,
            instruction=instruction,
        )
    
    elif subtype == "single":
        # For single: call Qwen first, then YOLO ensemble with max IOU selection
        LOGGER.info("Grounding subtype is 'single': calling Qwen first, then YOLO ensemble with max IOU")
        
        # Call Qwen fine-tuned grounding model
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url},
                        {"type": "text", "text": "You are a grounding and OBB (Oriented Bounding Box) detection model. Locate the object described. [refer] "+instruction}
                    ]
                }
            ]
            
            qwen_response = qwen_ft_ground(messages)
            LOGGER.info(f"Qwen grounding response: {qwen_response}")
            
            # Parse Qwen response
            qwen_obbs = parse_qwen_response(qwen_response)
            LOGGER.info(f"Parsed {len(qwen_obbs)} OBBs from Qwen response")
            
        except Exception as e:
            LOGGER.error(f"Failed to get Qwen grounding response: {e}")
            # Return empty list on error
            return []
        
        # Check if we have YOLO models for ensemble
        if yolo_model_1 is None and yolo_model_2 is None and yolo_base is None:
            LOGGER.info("YOLO models not available, using Qwen only")
            # Return Qwen results as-is
            results = []
            for idx, qwen_obb_dict in enumerate(qwen_obbs):
                results.append({
                    "object-id": str(idx + 1),
                    "obbox": qwen_obb_dict.get('obbox', [0.5, 0.5, 0.1, 0.1, 0.0])
                })
            return results
        
        # Use single object ensemble (which finds max IOU with Qwen)
        LOGGER.info("Using single object ensemble grounding with max IOU selection")
        return ensemble_single_object_grounding(
            qwen_obbs=qwen_obbs,
            image=image,
            yolo_model_1=yolo_model_1,
            yolo_model_2=yolo_model_2,
            yolo_base=yolo_base,
            object_class=object_class,
        )
    
    else:
        # Fallback: subtype is None or unknown - use original logic
        LOGGER.info(f"Grounding subtype is '{subtype}' or unknown: using original pipeline")
        
        # Call Qwen fine-tuned grounding model
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_url},
                        {"type": "text", "text": "You are a grounding and OBB (Oriented Bounding Box) detection model. Locate the object described. [refer] "+instruction}
                    ]
                }
            ]
            
            qwen_response = qwen_ft_ground(messages)
            LOGGER.info(f"Qwen grounding response: {qwen_response}")
            
            # Parse Qwen response
            qwen_obbs = parse_qwen_response(qwen_response)
            LOGGER.info(f"Parsed {len(qwen_obbs)} OBBs from Qwen response")
            
        except Exception as e:
            LOGGER.error(f"Failed to get Qwen grounding response: {e}")
            # Return empty list on error
            return []
        
        # Determine if we should use ensemble
        use_ensemble = (
            (yolo_model_1 is not None or yolo_model_2 is not None or yolo_base is not None) and
            len(qwen_obbs) > 0
        )
        
        if not use_ensemble:
            LOGGER.info("YOLO models not available or no Qwen predictions, using Qwen only")
            # Return Qwen results as-is
            results = []
            for idx, qwen_obb_dict in enumerate(qwen_obbs):
                results.append({
                    "object-id": str(idx + 1),
                    "obbox": qwen_obb_dict.get('obbox', [0.5, 0.5, 0.1, 0.1, 0.0])
                })
            return results
        
        # Use ensemble logic
        # Determine if single or multiple based on number of Qwen predictions
        is_single = len(qwen_obbs) == 1
        
        if is_single:
            LOGGER.info("Using single object ensemble grounding")
            return ensemble_single_object_grounding(
                qwen_obbs=qwen_obbs,
                image=image,
                yolo_model_1=yolo_model_1,
                yolo_model_2=yolo_model_2,
                yolo_base=yolo_base,
                object_class=object_class,
            )
        else:
            LOGGER.info("Using multi-object ensemble grounding")
            return ensemble_multi_object_grounding(
                qwen_obbs=qwen_obbs,
                image=image,
                yolo_model_1=yolo_model_1,
                yolo_model_2=yolo_model_2,
                yolo_base=yolo_base,
                object_class=object_class,
            )

