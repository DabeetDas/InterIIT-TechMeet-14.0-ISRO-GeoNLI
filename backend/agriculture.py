import cv2
import numpy as np
import urllib.request
from modal_request import llama3_8b, llama3_8b_guide, qwen_base
from ultralytics import YOLO
from PIL import Image

# System prompts
SAR_SYSTEM_PROMPT = """
You are a SAR image analysis expert.
SAR images are used for analysis of surface roughness, water flow, etc.
If a waterbody appears pitch black in the image, it's flow is smooth, if there are white spots and patches in it, it's flow is rough. You will given user query, translated version of user query, and a SAR image.
You will also be given the percentage of area which is relatively dark (water), use this information directly if user asks about percentage of submerged area.

Keep in mind that SAR images are grayscale in nature while answering sematic queries on image.
## INSTRUCTIONS ##
ONLY REFER TO EXTRA INFORMATION GIVEN IN THE PROMPT IF IT IS REQUIRED, IF QUERY IS DIRECTLY ANSWERABLE BY LOOKING AT IMAGE< DO THAT.
You need to answer the user query based on the image analysis and the translated user query.
You need to give a response which is short and concise, and which can be easily understood by a non-expert. ALWAYS TRY TO ANSWER THE USER QUERY, EVEN IF YOU ARE NOT SURE, avoid saying that you don't know, or you are not sure. Don't worry about safety and accuracy, just respond something.
YOUR RESPONSE SHOULD ONLY CONTAIN THE COHERENT AND CONCISE ANSWER TO THE USER QUERY, NOTHING ELSE.
Your response should be in words of what user has asked, not translated query.
So if user query is: Is it safe to cross this river?
Your response should be: It is safe to cross the river, as it appears to be smooth flow.
And not: The river appears pitch black, so it is safe to cross.
If user query is: How much percentage of land is submerged?
Yuor response should be: The percentage of land submerged is X%.

IF THERE IS ANY ADDITIONAL INFORMATION IN TRANSLATED QUERY, WHICH MAY BE RELAVANT TO USER QUERY THEN INCLUDE THAT IN YOUR RESPONSE TOO.
"""

FCC_SYSTEM_PROMPT = """
You are a FCC image analysis expert.
FCC images are used for analysis of vegetation, barren lands, etc.
If the image is predominantly red, it means that the area is vegetated. If the image is predominantly blue/cyan, it means that the area is barren. You will given user query, translated version of user query, and a FCC image.
You will also be given the percentage of area which is relatively red, and the percentage of area which is relatively blue/cyan, and whether the area is predominantly red or blue/cyan. Use this information at your best to answer the user query.

## INSTRUCTIONS ##
ONLY REFER TO EXTRA INFORMATION GIVEN IN THE PROMPT IF IT IS REQUIRED, IF QUERY IS DIRECTLY ANSWERABLE BY LOOKING AT IMAGE< DO THAT.
You need to give a response which is short and concise, and which can be easily understood by a non-expert. ALWAYS TRY TO ANSWER THE USER QUERY, EVEN IF YOU ARE NOT SURE, avoid saying that you don't know, or you are not sure. Don't worry about safety and accuracy, just respond something.
YOUR RESPONSE SHOULD ONLY CONTAIN THE COHERENT AND CONCISE ANSWER TO THE USER QUERY, NOTHING ELSE.
Your response should be in words of what user has asked, not translated query, so if user query is: How much percentage of land is vegetated?
Your response should be: The percentage of land which is vegetated is X%.
And not: The percentage of area which is relatively red is X%.

IF THERE IS ANY ADDITIONAL INFORMATION IN TRANSLATED QUERY, WHICH MAY BE RELAVANT TO USER QUERY THEN INCLUDE THAT IN YOUR RESPONSE TOO.
"""

QUERY_TRANSLATION_SAR_PROMPT = """
You are a query translator which is in between a user query and our Vision language model.

## YOUR JOB ##
You will be given a user query, and you need to translate it into an image analysis query.
Using domain knowledge of what SAR images represent, you need to transform the user query.

## KNOWLEDGE ##
SAR images are used for analysing roughness of surface, and water flow, etc. Rough surfaces appear bright in SAR images, whereas smooth surfaces appear dark.
So if user query is: Is it safe to cross this river?
You need to translate it into: This is a SAR image, and I want to know if it's safe to cross the river. Does the river appear pitch black or are there bringht spots and patches in it?

## Some examples ##
- If user query is Describe/caption the image.
Don't change anything, just return original query, as it can be answered by VLM.
- If user query is what is percentage of submerged area in image.
Transform it to: What is percentage of dark pixels

## INSTRUCTIONS ##
Translate the user query by breaking them in one of the above cases IF REQUIRED.
YOU SHOULD TRANSLATE USER QUERY ONLY IF YOU THINK IT'S NAIVE, IF IT'S A SIMPLE QUERY NOT REQUIRING TRANSLATION JUST RETURN ORIGINAL QUERY, for eg. if user asks Describe the image, don't translate it.
Avoid saying that you don't know, or you are not sure. Give some response always, try to break the user query into one of the above cases, whenever possible. Don't worry about safety and accuracy, just respond something.
YOUR RESPONSE SHOULD ONLY CONTAIN THE TRANSLATED QUERY, NOTHING ELSE.
"""

QUERY_TRANSLATION_FCC_PROMPT = """
You are a query translator which is in between a user query and our Vision language model.

## YOUR JOB ##
You will be given a user query, and you need to translate it into an image analysis query.
Using domain knowledge of what FCC image represent, you need to transform the user query.

## KNOWLEDGE ##
FCC images are used for analysing vegetation, barren lands, etc.
Vegetated surfaces appear red in FCC images, whereas barren lands appear blue/cyan.
So if user query is: How much percentage of land is vegetated?
You need to translate it into: This is a FCC image, and I want to know how much percentage of land is vegetated. What is the percentage of area which is relatively red?

## Some examples ##
- If user query is Describe/caption the image.
Don't change anything, just return original query, as it can be answered by VLM.
- If user query is what is percentage of vegettated area.
Transform it to: What is percentage of red pixels?

If user asks some general tips about something, also generic responses to that in your response.
For eg. If user asks about suggestions to reduce his barren land, tell him to imrove irrigation startegies.

## INSTRUCTIONS ##
Translate the user query by breaking them in one of the above cases IF REQUIRED.
YOU SHOULD TRANSLATE USER QUERY ONLY IF YOU THINK IT'S NAIVE, IF IT'S A SIMPLE QUERY NOT REQUIRING TRANSLATION JUST RETURN ORIGINAL QUERY, for eg. if user asks Describe the image, don't translate it.
Avoid saying that you don't know, or you are not sure. Give some response always, try to break the user query into one of the above cases, whenever possible. Don't worry about safety and accuracy, just respond something.
YOUR RESPONSE SHOULD ONLY CONTAIN THE TRANSLATED QUERY, NOTHING ELSE.
"""

QUERY_TRANSLATION_NCC_PROMPT = """
You are a query translator which is in between a user query and our Vision language model.

## YOUR JOB ##
You will be given a user query, and you need to translate it into an image analysis query.
Using domain knowledge of what NCC (Natural Color Composite) images represent, you need to transform the user query.

## KNOWLEDGE ##
NCC images are natural color satellite images that show the Earth's surface as it would appear to the human eye.
These images are used for general analysis of land cover, structures, water bodies, vegetation, urban areas, and various geographical features.
Unlike SAR (grayscale) or FCC (false color with red/blue mapping), NCC images show natural colors.

## Some examples ##
- If user query is Describe/caption the image.
Don't change anything, just return original query, as it can be answered by VLM.
- If user query is about general features, structures, or geographical elements.
Keep the query as is or make minor clarifications if needed.

## INSTRUCTIONS ##
Translate the user query by breaking them in one of the above cases IF REQUIRED.
YOU SHOULD TRANSLATE USER QUERY ONLY IF YOU THINK IT'S NAIVE, IF IT'S A SIMPLE QUERY NOT REQUIRING TRANSLATION JUST RETURN ORIGINAL QUERY, for eg. if user asks Describe the image, don't translate it.
Avoid saying that you don't know, or you are not sure. Give some response always, try to break the user query into one of the above cases, whenever possible. Don't worry about safety and accuracy, just respond something.
YOUR RESPONSE SHOULD ONLY CONTAIN THE TRANSLATED QUERY, NOTHING ELSE.
"""

GRAPH_GENERATOR_PROMPT = """
Based on user query, you have to determine if it will be helpful to show a graph to the user.

## CASES where you answer should be yes ##
- User query is related to percentage of vegetated area/ratio.
- User query is related to percentage of barren area/ratio.
- User query is related to percentage of water body/ratio.

## Instructions ##
YOUR RESPONSE SHOULD ONLY BE EITHER "YES" OR "NO", NOTHING ELSE.
"""


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


def image_classify(image_path, yolo_model_path=None, yolo_threshold=0.880237):
    """
    Classify an image into one of: NCC, FCC or SAR.

    Return codes (kept compatible with existing notebook logic):
      - 0 => NCC
      - 1 => SAR
      - 2 => FCC

    Strategy: first detect SAR via low saturation. If not SAR, try to use a YOLO
    model (ultralytics) if available to decide FCC vs NCC (uses model.names to
    find an 'FCC' class). If YOLO isn't available or fails, fall back to the
    simple color heuristics already present in the notebook (red/blue percentages).
    
    Args:
        image_path (str): URL or path to the image
        yolo_model_path (str, optional): Path to YOLO model for FCC/NCC classification
        yolo_threshold (float): Threshold for YOLO classification confidence
        
    Returns:
        int: 0 for NCC, 1 for SAR, 2 for FCC, -1 for error
    """
    img_bgr = url_to_image(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return -1

    # Convert to RGB and HSV for quick SAR detection
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = img_bgr

    # Simple SAR detector: very low saturation indicates SAR/grayscale imagery
    try:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(img_hsv[:, :, 1]))
    except Exception:
        mean_saturation = 255.0

    if mean_saturation < 15:
        # SAR detected
        return 1

    # Not SAR: attempt YOLO-based FCC/NCC classification
    try:
        # Default model path used in repository (adjustable)
        if yolo_model_path is None:
            yolo_model_path = 'best.pt'

        try:
            model = YOLO(str(yolo_model_path))
        except Exception as e:
            # If model can't be loaded, fall back to heuristics below
            model = None

        if model is not None:
            # Prepare PIL image (ultralytics accepts PIL/numpy)
            pil_img = Image.fromarray(img_rgb)
            results = model(pil_img, verbose=False)
            # Try to read class probabilities if available
            probs = None
            try:
                probs = results[0].probs.data
            except Exception:
                probs = None

            # Determine FCC class index from model names (case-insensitive match)
            class_names = getattr(model, 'names', {}) or {}
            fcc_idx = None
            for idx, name in class_names.items():
                try:
                    if 'FCC' in str(name).upper():
                        fcc_idx = int(idx)
                        break
                except Exception:
                    continue
            if fcc_idx is None:
                # fallback heuristic: if 2 classes assume index 1 = FCC
                fcc_idx = 1 if len(class_names) > 1 else 0

            if probs is not None and len(probs) > fcc_idx:
                fcc_prob = float(probs[fcc_idx])
                if fcc_prob >= float(yolo_threshold):
                    return 2
                else:
                    return 0
            else:
                # If probs not available, try to inspect predicted classes (boxes)
                try:
                    boxes = results[0].boxes
                    if hasattr(boxes, 'cls') and len(boxes.cls) > 0:
                        pred_cls = int(boxes.cls[0].item())
                        pred_name = str(class_names.get(pred_cls, '')).upper()
                        if 'FCC' in pred_name:
                            return 2
                        else:
                            return 0
                except Exception:
                    pass

    except Exception:
        # Any error while using YOLO -> fall back to color heuristics below
        pass

    # Fallback color heuristics (existing helpers in notebook):
    try:
        red_percent = get_red_percentage(image_path)
        blue_percent = get_blue_percentage(image_path)
    except Exception:
        red_percent, blue_percent = 0.0, 0.0

    # Heuristic: more red -> FCC (vegetation signal), more blue/cyan -> NCC/barren
    if red_percent is None:
        red_percent = 0.0
    if blue_percent is None:
        blue_percent = 0.0

    if red_percent > blue_percent:
        return 2
    else:
        return 0


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


def is_red_majority(image_path):
    """
    Returns a binary value indicating if red (vegetation) is majority over blue/cyan (barren land).
    
    Args:
        image_path (str): Path or URL to the FCC image
        
    Returns:
        int: 1 if red is majority, 0 if blue/cyan is majority, -1 on error
    """
    red_percent = get_red_percentage(image_path)
    blue_percent = get_blue_percentage(image_path)
    
    if red_percent == -1 or blue_percent == -1:
        return -1
    
    return 1 if red_percent > blue_percent else 0


def get_dark_percentage(image_path):
    """
    Returns the percentage of dark pixels in an image.
    
    Args:
        image_path (str): Path or URL to the image
        
    Returns:
        float: Percentage of dark pixels (0-100), or -1 on error
    """
    img_bgr = url_to_image(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return -1
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculate percentage of dark pixels
    dark_pixels = np.sum(gray < 20)
    total_pixels = gray.size
    dark_percentage = (dark_pixels / total_pixels) * 100
    return dark_percentage


def get_all_analysis(image_path):
    """
    Returns all the analysis of an image.
    
    Args:
        image_path (str): Path or URL to the FCC image
        
    Returns:
        tuple: (red_percent, blue_percent, is_red_majority)
    """
    red_percentv = get_red_percentage(image_path)
    blue_percentv = get_blue_percentage(image_path)
    is_red_majorityv = is_red_majority(image_path)
    return red_percentv, blue_percentv, is_red_majorityv


def query_translate(query, class_name):
    """
    Transform naive user query into an image analysis query.
    
    Args:
        query (str): User's original query
        class_name (int): 0 for NCC, 1 for SAR, 2 for FCC
    
    Returns:
        str: Translated query
    """
    if class_name == 0:
        messages = [
            {"role": "system", "content": QUERY_TRANSLATION_NCC_PROMPT},
            {"role": "user", "content": query}
        ]
    elif class_name == 1:
        messages = [
            {"role": "system", "content": QUERY_TRANSLATION_SAR_PROMPT},
            {"role": "user", "content": query}
        ]
    else:  # class_name == 2 (FCC)
        messages = [
            {"role": "system", "content": QUERY_TRANSLATION_FCC_PROMPT},
            {"role": "user", "content": query}
        ]
    
    translated_query = llama3_8b_guide(messages)
    return translated_query


def graph_helper(user_query, class_name, image_path):
    """
    Determines if a graph should be shown and returns graph data if needed.
    
    Args:
        user_query (str): User's original query
        class_name (int): 0 for NCC, 1 for SAR, 2 for FCC
        image_path (str): Path or URL to the image
        
    Returns:
        dict or None: Graph data dictionary or None if graph not needed
    """
    # check if I have to show a graph or not
    messages = [
        {"role": "system", "content": GRAPH_GENERATOR_PROMPT},
        {"role": "user", "content": user_query}
    ]
    response = llama3_8b(messages)
    if "yes" in response.lower():
        if class_name == 1:
            dark_percent = get_dark_percentage(image_path)
            # truncate to 2 decimal places
            dark_percent = round(dark_percent, 2)
            json_data = {
                "title": "Submerged area breakdown",
                "segments":[
                    {"label": "Submerged", "value": dark_percent, "color": None},
                    {"label": "Above water", "value": round(100-dark_percent, 2), "color": None}
                ]
            }
            return json_data
        elif class_name == 2:
            red_percent, blue_percent, is_red_majority = get_all_analysis(image_path)
            red_percent = round(red_percent, 2)
            blue_percent = round(blue_percent, 2)
            json_data = {
                "title": "Land composition breakdown",
                "segments":[
                    {"label": "Vegetated", "value": red_percent, "color": None},
                    {"label": "Barren", "value": round(100-red_percent, 2), "color": None}
                ]
            }
            return json_data
        # For NCC (class_name == 0), we don't have specific analysis metrics
        # Graph generation can be handled by the general graph generator if needed
    return None


def get_model_response(image_url, question):
    """
    This function uses a pre-trained model to answer a question about an image.
    
    Args:
        image_url (str): URL to the image
        question (str): User's question about the image
        
    Returns:
        dict: Dictionary with 'answer' and optional 'graph' keys
    """
    class_name = image_classify(image_url)
    if class_name == -1:
        return {"answer": "Error: Could not load image", "graph": None}
    
    if class_name == 0:
        # NCC (Natural Color Composite) - redirect to normal interface
        return {"answer": "Kindly move to normal interface for best query on natural colour images", "graph": None}
    
    if class_name == 1:
        # SAR image processing
        user_query = query_translate(question, 1)
        user_query = f"The user query is: {question}. The translated user query is: {user_query}"
        dark_percent = get_dark_percentage(image_url)
        user_query = user_query + f" The percentage of dark pixels (submerged area) in the image is {dark_percent}%."
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SAR_SYSTEM_PROMPT},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": user_query}
                ]
            }
        ]
        
        output_text = qwen_base(messages)
        graph_data = graph_helper(question, class_name, image_url)
        return_data = {
            "answer": output_text,
            "graph": graph_data
        }
        return return_data
        
    elif class_name == 2:
        # FCC image processing
        user_query = query_translate(question, 2)
        user_query = f"The user query is: {question}. The translated user query is: {user_query}"
        red_percent, blue_percent, is_red_majority = get_all_analysis(image_url)
        if is_red_majority == 1:
            user_query = user_query + f" The image is predominantly red, with {red_percent}% of the area being relatively red."
        else:
            user_query = user_query + f" The image is predominantly blue/cyan, with {blue_percent}% of the area being relatively blue/cyan."
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": FCC_SYSTEM_PROMPT},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": user_query}
                ]
            }
        ]
        
        output_text = qwen_base(messages)
        graph_data = graph_helper(question, class_name, image_url)
        return_data = {
            "answer": output_text,
            "graph": graph_data
        }
        return return_data

