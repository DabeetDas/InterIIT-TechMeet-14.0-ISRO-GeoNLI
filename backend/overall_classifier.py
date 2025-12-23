"""
Overall classifier module for VRSBench pipeline.

This module classifies user queries to determine the appropriate task type
and routing to the correct model.
"""

import json
import re
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from modal_request import llama3_8b

# Known object classes for VRSBench-obb
KNOWN_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
    "windmill", "dam", "trainstation", "overpass", "stadium", "airport", "helipad",
    "golffield", "chimney", "expressway service area", "expressway toll station",
    "container crane"
]

# Define the classification schema using Pydantic
class QueryClassification(BaseModel):
    """Classification result for a user query"""
    
    task_type: Literal[
        "captioning",
        "grounding", 
        "vqa_float",
        "semantic",
        "binary",
        "general"
    ] = Field(
        description="The main task type for the query."
    )
    
    grounding_subtype: Optional[Literal["single", "multiple"]] = Field(
        default=None,
        description="If task_type is 'grounding', specify 'single' for single object bbox or 'multiple' for multiple objects bbox. None otherwise."
    )
    
    vqa_float_subtype: Optional[Literal["counting", "area", "perimeter", "length", "other"]] = Field(
        default=None,
        description="If task_type is 'vqa_float', specify 'counting' for object counting queries, 'area' for area measurements, 'perimeter' for perimeter measurements, 'length' for linear measurements (length, width, height), or 'other' for other numerical answers. None otherwise."
    )
    
    object_class: Optional[str] = Field(
        default=None,
        description="The specific object or feature name targeted by the query. First, check if it matches the KNOWN_CLASSES list exactly. If it does not match (e.g. 'river', 'road'), you MUST return the object name as a string extracted from the query. Do not return null unless the query is completely generic (e.g. 'what is the area?')."
    )

# Build system prompt
KNOWN_CLASSES_STR = "\n".join([f"  {i}: {cls}" for i, cls in enumerate(KNOWN_CLASSES)])

SYSTEM_PROMPT = """You are an expert query classifier for a remote sensing satellite imagery understanding system. This system processes and analyzes satellite/aerial images to answer questions about objects, features, and content visible in the imagery.

**IMPORTANT CONTEXT**: All queries are related to analyzing remote sensing satellite imagery data. The system is designed to answer questions about objects, features, and content visible in satellite images.

You MUST respond with ONLY a valid JSON object, no additional text before or after.

Available task types and their descriptions:

1. **captioning**: User wants a comprehensive image description or caption of the satellite imagery. This includes requests like "describe this image", "what's in this picture", "caption this image", "give me a detailed description of this satellite image", etc. The model used is a fine-tuned Qwen model specifically trained for generating detailed captions of remote sensing imagery.

2. **grounding**: User wants bounding box localization of objects in the satellite imagery. This requires detecting and localizing where specific objects are located in the image.

   - **single**: User wants a single object's bounding box (e.g., "where is the plane?", "show me the ship", "locate the storage tank")

   - **multiple**: User wants multiple objects' bounding boxes (e.g., "find all ships", "locate all vehicles", "show me all planes")

   - Uses specialized grounding models (single object or multi-object) trained for object detection in satellite imagery.

3. **vqa_float**: User wants a numerical/float answer about the satellite imagery. This includes counting objects or measuring specific dimensions.

   - **counting**: User wants to count objects visible in the satellite image (e.g., "how many planes are in the image?", "count the number of ships", "how many vehicles can you see?")

   - **area**: User wants the area measurement of an object or feature (e.g., "what is the area of the building?", "calculate the area of the lake").

   - **perimeter**: User wants the perimeter measurement of an object or feature (e.g., "what is the perimeter of the harbor?", "measure the perimeter of the field").

   - **length**: User wants linear measurements such as length, width, or height of features (e.g., "how long is the road?", "what is the width of the bridge?", "how tall is the structure?").

   - **other**: User wants a numerical answer that does not fit counting, area, perimeter, or length (e.g., "what is the cloud cover percentage?").

4. **semantic**: User wants a short semantic answer (1-5 words) about objects, features, or content visible in the satellite imagery. These are questions that ask for brief identification or classification of what is visible in the image.

   - **CRITICAL**: This is ONLY for questions about objects, features, or content visible in the satellite imagery itself.

   - Examples: "what type of vehicle is this?" (referring to a vehicle in the image), "which object is this?" (referring to something in the image), "what kind of building is this?" (about a building in the image), "what is this structure?" (about a structure visible in the satellite image)

   - Uses a fine-tuned Qwen model trained to give 1-5 word answers about remote sensing imagery content.

   - **NOT for general questions**: Questions about the system itself, creators, general knowledge, or anything not directly related to analyzing the satellite imagery should NOT be classified as semantic.

5. **binary**: User wants a yes/no answer about the presence or characteristics of objects/features in the satellite imagery. These are verification questions about what is visible in the image.

   - Examples: "is there a plane in this image?", "does this contain a ship?", "are there any vehicles visible?", "is this an airport?"

   - Uses a specialized binary model for yes/no answers about satellite imagery content.

6. **general**: General queries that are NOT about analyzing the satellite imagery itself. This includes:

   - Questions about the system itself (e.g., "what is name of your creators?", "who made you?", "what is your purpose?")

   - General conversation not related to image analysis

   - Questions about general knowledge that don't relate to analyzing the current satellite image

   - Requests for explanations about how the system works

   - Any query that doesn't fit into the above categories or is not asking about content visible in the satellite imagery

   - Uses the base Qwen model for general text generation and conversation.

**Known Object Classes for VRSBench-obb:**

""" + KNOWN_CLASSES_STR + """

**Important Guidelines:**

- **CRITICAL**: This system is for analyzing remote sensing satellite imagery. All task types (except "general") are for questions about content visible in satellite images.

- Analyze the conversation history to understand context

- Consider the intent behind the current query - is it asking about the satellite image content or something else?

- **Semantic vs General distinction**:

  - **semantic**: Questions asking "what is this?" or "which object?" that refer to objects/features visible in the satellite imagery (e.g., "what type of vehicle is this?" about a vehicle in the image)

  - **general**: Questions about the system itself, creators, general knowledge, or anything NOT about analyzing the satellite imagery (e.g., "what is name of your creators?", "who made you?", "explain how you work")

- If a query asks for bounding boxes or object localization in the satellite image, classify as "grounding"

- If a query asks for numerical answers (numbers, measurements) about objects/features in the satellite image, classify as "vqa_float"

- If a query is asking for a brief identification (1-5 words) about objects/features visible in the satellite imagery, classify as "semantic"

- If a query is asking for yes/no about presence of objects/features in the satellite image, classify as "binary"

- If a query is asking for image description/captioning of the satellite imagery, classify as "captioning"

- For grounding queries, determine if user wants single or multiple objects based on keywords like "all", "each", "every", vs singular forms

- For vqa_float queries, you MUST distinguish the specific subtype: "counting", "area", "perimeter", "length", or "other".

**Object Class Identification:**

- For "grounding" task type: You MUST identify if the query is asking about one of the known classes listed above. If yes, set "object_class" to the matching class name (use exact name from the list). If the query is about an object not in the list, you MUST provide the best possible name for this object based on the query (not necessarily from the known classes list). If the query is generic (not about a specific object), set "object_class" to null.

- For "vqa_float" task type with subtypes "counting", "area", "perimeter", or "length": You MUST identify if the query is asking about one of the known classes. If yes, set "object_class" to the matching class name (use exact name from the list). If not in the list, provide the best possible name for this object based on the query (not necessarily from the known classes list). If there is no specific object, set "object_class" to null.

- For all other task types, set "object_class" to null.

**Response Format (JSON only):**

{
    "task_type": "one of: captioning, grounding, vqa_float, semantic, binary, general",
    "grounding_subtype": "single or multiple (only if task_type is grounding, else null)",
    "vqa_float_subtype": "one of: counting, area, perimeter, length, other (only if task_type is vqa_float, else null)",
    "object_class": "Identify the target subject of the query. Follow this strict logic:
                     1. If the subject matches a name in 'Known Object Classes', use that exact name.
                     2. If the subject is NOT in the list (e.g., 'river', 'forest', 'road', 'building'), YOU MUST EXTRACT the noun used in the query.
                     3. Only return null if the query mentions no specific object (e.g., 'what is the area of everything?')."
}"""


def classify_query_from_messages(messages: List[dict]) -> QueryClassification:
    """
    Classify a user query from a messages array.
    
    The messages array should be in the format:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
    
    The last user message is treated as the current query to classify.
    
    Args:
        messages: List of conversation messages in format [{"role": "...", "content": "..."}]
    
    Returns:
        QueryClassification object with task_type, subtypes, and object_class
    """
    if not messages:
        raise ValueError("Messages array cannot be empty")
    
    # Extract conversation history and current query
    conversation_history = []
    current_query = None
    
    # Process messages to separate history from current query
    # The last user message is the current query
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Handle different content formats
        if isinstance(content, list):
            # For multimodal messages, extract text content
            text_content = ""
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        break
            content = text_content
        elif not isinstance(content, str):
            content = str(content)
        
        if role == "user":
            if i == len(messages) - 1:
                # Last user message is the current query
                current_query = content
            else:
                # Previous user messages are part of history
                conversation_history.append({"role": "user", "content": content})
        elif role == "assistant":
            conversation_history.append({"role": "assistant", "content": content})
    
    # If no current query found, use the last message
    if current_query is None:
        last_msg = messages[-1]
        content = last_msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    current_query = item.get("text", "")
                    break
        else:
            current_query = str(content)
    
    if not current_query:
        raise ValueError("Could not extract current query from messages")
    return classify_query(current_query, conversation_history)


def classify_query(
    current_query: str,
    conversation_history: Optional[List[dict]] = None
) -> QueryClassification:
    """
    Classify a user query based on conversation history and current query.
    Uses the local Llama model to get the classification result.
    
    Args:
        current_query: The current user query to classify
        conversation_history: Optional list of conversation messages in format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Returns:
        QueryClassification object with task_type, subtypes, and object_class
    """
    # Build messages for the pipeline
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
    # Add the current query
    user_prompt = f"Current query: {current_query}\n\nClassify this query and respond with ONLY the JSON object as specified. Include object_class only if task_type is grounding or if task_type is vqa_float with subtypes counting, area, perimeter, or length."
    messages.append({"role": "user", "content": user_prompt})
    
    # Get classification from model (using Modal llama3_8b)
    try:
        response_text = llama3_8b(messages)
    except Exception as e:
        raise ValueError(f"Failed to get classification from model: {str(e)}")
    
    # Extract JSON from response (handle cases where LLM adds extra text)
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response_text
    
    # Parse JSON and create QueryClassification object
    try:
        result_dict = json.loads(json_str)
        # Handle null values in JSON (convert to None for Python)
        if result_dict.get("grounding_subtype") is None:
            result_dict["grounding_subtype"] = None
        if result_dict.get("vqa_float_subtype") is None:
            result_dict["vqa_float_subtype"] = None
        if result_dict.get("object_class") is None or result_dict.get("object_class") == "null":
            result_dict["object_class"] = None
        
        # Validate object_class if present - must be in known classes or None
        if result_dict.get("object_class") and result_dict["object_class"] not in KNOWN_CLASSES:
            # Try to find a close match (case-insensitive)
            obj_class_lower = result_dict["object_class"].lower()
            matched = None
            for known_class in KNOWN_CLASSES:
                if known_class.lower() == obj_class_lower:
                    matched = known_class
                    break
            if matched:
                result_dict["object_class"] = matched
            else:
                if result_dict.get("task_type") == "grounding":
                    result_dict["object_class"] = None
        
        classification = QueryClassification(**result_dict)
    except (json.JSONDecodeError, Exception) as e:
        # Fallback: try to construct from text if JSON parsing fails
        print(f"Warning: JSON parsing failed, attempting to extract information. Error: {e}")
        print(f"Response text: {response_text}")
        raise ValueError(f"Failed to parse classification response: {response_text}")
    
    return classification


# Helper function to get model selection based on classification
def get_model_from_classification(classification: QueryClassification) -> str:
    """
    Map classification result to the appropriate model to use.
    
    Returns:
        String indicating which model to use
    """
    task = classification.task_type
    
    if task == "captioning":
        return "captioning_model"  # fine-tuned qwen for caption generation
    
    elif task == "grounding":
        if classification.grounding_subtype == "multiple":
            return "grounding_multi_objects_model"
        else:
            return "grounding_single_object_model"
    
    elif task == "vqa_float":
        if classification.vqa_float_subtype == "counting":
            return "vqa_counting_model"
        elif classification.vqa_float_subtype in ["area", "perimeter", "length"]:
            return "vqa_size_model"  # calculates masks for size measurements
        else:  # other
            return "vqa_other_model"  # base qwen for other numeric queries
    
    elif task == "semantic":
        return "semantic_model"  # fine-tuned qwen for 1-5 word answers
    
    elif task == "binary":
        return "binary_model"  # yes/no answer model
    
    else:  # general
        return "base_qwen_model"  # base qwen for general queries

