from modal_request import llama3_8b_guide, qwen_base
from typing import Dict, Any, Optional, Tuple
import time
# Error fallback response
ERROR_RESPONSE = {
    "caption": "something went wrong",
    "llama_guide_text": "something went wrong",
    "qwen_description": "something went wrong",
    "image_type": "SAR",
    "error": True
}

def build_qwen_caption_prompts(image_type: str) -> Tuple[str, str]:
    """
    Build system and user prompts for Qwen caption generation.
    
    Args:
        image_type: "SAR" or "FCC"
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if image_type == "SAR":
        system = "You are an expert in remote sensing image interpretation with specialization in Synthetic Aperture Radar (SAR). Provide detailed, accurate descriptions in professional terms."
        user = (
            "This is a SAR image (grayscale). Describe the picture of a remote sensing scene in approximately 60 words. "
            "Focus on SAR-specific cues:\n"
            "- Backscatter characteristics: dark patches, bright spots, linear stretches, speckle, textural differences.\n"
            "- Land cover indicators: smooth water bodies (dark), urban areas (bright point scatterers), vegetation textures (medium), bare soil, rocky terrain.\n"
            "- Geometric and structural features: linear infrastructures (roads, railways), grid-like urban patterns, facilities.\n"
            "- Shadows and layover: angular bright returns, shadow regions indicating tall structures or steep slopes.\n"
            "- Spatial context: relative positions, clustering, gradients, boundaries.\n"
            "- Anomalies or notable patterns: repeated motifs, isolated bright/dark targets, ridge lines.\n"
            "Provide a neutral, technical description without speculation beyond visual evidence."
        )
    else:  # FCC
        system = "You are an expert in remote sensing image interpretation with specialization in NIR false color composites (FCC). Provide detailed, accurate descriptions in professional terms."
        user = (
            "This is an FCC image (color). Describe the picture of a remote sensing scene in approximately 60 words. "
            "Focus on FCC-specific cues:\n"
            "- Vegetation vigor (red tones), moisture gradients, crop variability, forest canopy textures.\n"
            "- Non-vegetated surfaces: built-up areas (cyan/gray), bare soil (tan/brown), rock (gray), water (dark/black/blue depending on composite).\n"
            "- Color contrasts and boundaries, patchiness, linear features (roads, canals), field parcel shapes.\n"
            "- Spatial arrangement and land use indicators: settlements, industrial zones, agricultural mosaics, wetlands.\n"
            "- Indicators of change or anomalies: unusually pale/red areas, mixed pixels, turbidity in water bodies.\n"
            "Provide a neutral, technical description without speculation beyond visual evidence."
        )
    return system, user


def build_llama_summary_prompt(qwen_description: str) -> Tuple[str, str]:
    """
    Build system and user prompts for Llama to summarize Qwen description.
    
    Args:
        qwen_description: Detailed description from Qwen
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = "You are a remote sensing knowledge expert. Summarize the following domain-specific image description into a concise caption of about 60 words. Emphasize the most distinctive and informative elements for SAR/FCC interpretation. Keep it factual and neutral."
    user = f"Summarize into 60 words:\n{qwen_description}"
    return system, user


def build_llama_vqa_guidance_prompt(image_type: str, question: str) -> Tuple[str, str]:
    """
    Build system and user prompts for Llama to generate VQA guidance.
    
    Args:
        image_type: "SAR" or "FCC"
        question: The VQA question
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = "You are a remote sensing domain coach. Provide a short checklist of cues and clues the vision model should focus on to answer the question on a remote sensing image. Keep it under 60 words and specific to the question."
    user = (
        f"Image type: {image_type}. Question: {question}.\n"
        "- Return ONLY a focused checklist of visual cues (no extra prose).\n"
        "- Mention what to look for, where to look, which attributes (texture, backscatter/color), and likely confounders.\n"
    )
    return system, user


def build_qwen_vqa_prompt(image_type: str, llama_guidance: str, question: str, caption: Optional[str] = None) -> Tuple[str, str]:
    """
    Build system and user prompts for Qwen VQA with guidance and caption.
    
    Args:
        image_type: "SAR" or "FCC"
        llama_guidance: Guidance text from Llama
        question: The VQA question
        caption: Optional caption from previous step
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = (
        "You are an expert vision-language model for remote sensing images. You must analyze the provided IMAGE to answer."
        " Use the guidance and caption strictly as hints. Do NOT repeat them verbatim. Do NOT restate system/user messages."
        " Answer the question directly and ground your answer in visual evidence."
        " Format: 'Answer: <concise answer in 1–2 sentences>'."
    )
    caption_block = f"Caption (context; do not repeat):\n{caption}\n\n" if caption else ""
    user = (
        f"Image type: {image_type}\n"
        f"{caption_block}"
        f"Guidance (hints; do not repeat):\n{llama_guidance}\n\n"
        f"Question: {question}\n"
        "Requirements:\n"
        "- Look at the attached IMAGE and base your answer primarily on what is visible.\n"
        "- Use the caption and guidance only to focus attention; do not echo them.\n"
        "- Do not output any roles, headers, or the guidance/caption text.\n"
        "- If uncertain, say 'Answer: Uncertain' and briefly state why.\n"
        "Answer now."
    )
    return system, user


def sar(image_url: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate caption for SAR image using Qwen and Llama pipeline.
    
    Args:
        image_url: URL or path to the SAR image
        prompt: Optional custom prompt (if None, uses default SAR caption prompt)
        
    Returns:
        JSON dict with:
        - caption: Final summarized caption (60 words)
        - llama_guide_text: Placeholder for VQA guidance (generated when question provided)
        - qwen_description: Detailed description from Qwen (60 words)
        - image_type: "SAR"
        - error: False if successful, True if error occurred
    """
    try:
        # Build Qwen prompts for SAR
        system_prompt, user_prompt = build_qwen_caption_prompts("SAR")
        
        # Use custom prompt if provided
        if prompt:
            user_prompt = prompt
        
        # Build messages for Qwen (with image)
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # Generate detailed description with Qwen
        print("Generating detailed description with Qwen")
        start_time = time.time()
        print(f"messages passed to qwen_base: {messages}")
        qwen_description = qwen_base(messages)
        end_time = time.time()
        print(f"Time taken to generate detailed description with Qwen: {end_time - start_time} seconds")
        
        if not qwen_description or qwen_description.strip() == "":
            error_response = ERROR_RESPONSE.copy()
            error_response["image_type"] = "SAR"
            return error_response
        
        # Build Llama summary prompts
        llama_system, llama_user = build_llama_summary_prompt(qwen_description)
        
        # Build messages for Llama (text-only)
        llama_messages = [
            {"role": "system", "content": llama_system},
            {"role": "user", "content": llama_user}
        ]
        
        # Generate caption with Llama
        print("Generating caption with Llama")
        start_time = time.time()
        print(f"messages passed to llama3_8b_guide: {llama_messages}")
        caption = llama3_8b_guide(llama_messages)
        end_time = time.time()
        print(f"Time taken to generate caption with Llama: {end_time - start_time} seconds")
        
        if not caption or caption.strip() == "":
            error_response = ERROR_RESPONSE.copy()
            error_response["image_type"] = "SAR"
            return error_response
        
        # Return JSON with all relevant info
        return {
            "caption": caption.strip(),
            "llama_guide_text": "",  # Will be generated when VQA question is provided
            "qwen_description": qwen_description.strip(),
            "image_type": "SAR",
            "error": False
        }
        
    except Exception as e:
        print(f"Error in sar function: {str(e)}")
        error_response = ERROR_RESPONSE.copy()
        error_response["image_type"] = "SAR"
        return error_response


def fcc(image_url: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate caption for FCC image using Qwen and Llama pipeline.
    
    Args:
        image_url: URL or path to the FCC image
        prompt: Optional custom prompt (if None, uses default FCC caption prompt)
        
    Returns:
        JSON dict with:
        - caption: Final summarized caption (60 words)
        - llama_guide_text: Placeholder for VQA guidance (generated when question provided)
        - qwen_description: Detailed description from Qwen (60 words)
        - image_type: "FCC"
        - error: False if successful, True if error occurred
    """
    try:
        # Build Qwen prompts for FCC
        system_prompt, user_prompt = build_qwen_caption_prompts("FCC")
        
        # Use custom prompt if provided
        if prompt:
            user_prompt = prompt
        
        # Build messages for Qwen (with image)
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # Generate detailed description with Qwen
        qwen_description = qwen_base(messages)
        
        if not qwen_description or qwen_description.strip() == "":
            error_response = ERROR_RESPONSE.copy()
            error_response["image_type"] = "FCC"
            return error_response
        
        # Build Llama summary prompts
        llama_system, llama_user = build_llama_summary_prompt(qwen_description)
        
        # Build messages for Llama (text-only)
        llama_messages = [
            {"role": "system", "content": llama_system},
            {"role": "user", "content": llama_user}
        ]
        
        # Generate caption with Llama
        caption = llama3_8b_guide(llama_messages)
        
        if not caption or caption.strip() == "":
            error_response = ERROR_RESPONSE.copy()
            error_response["image_type"] = "FCC"
            return error_response
        
        # Return JSON with all relevant info
        return {
            "caption": caption.strip(),
            "llama_guide_text": "",  # Will be generated when VQA question is provided
            "qwen_description": qwen_description.strip(),
            "image_type": "FCC",
            "error": False
        }
        
    except Exception as e:
        print(f"Error in fcc function: {str(e)}")
        error_response = ERROR_RESPONSE.copy()
        error_response["image_type"] = "FCC"
        return error_response
