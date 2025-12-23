from modal_request import qwen_ft_caption

SYSTEM_PROMPT = """
Examine the satellite image and provide a concise, precise, and fully objective caption.
Describe only what is clearly visible: land cover, structures, roads, water bodies, vegetation, and notable spatial patterns.
Use clear, coherent wording with no filler or repetition.
Avoid speculation or interpretation.
Keep the caption as short and information-focused as possible.
Answer in around 55 words.
"""

def caption(image_url, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":[
            {"type": "image", "image":image_url},
            {"type": "text", "text":prompt}
        ]}
    ]
    response = qwen_ft_caption(messages)
    return response