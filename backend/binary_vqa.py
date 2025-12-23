import re
from modal_request import qwen_ft_caption

def _parse_binary_answer(text: str) -> str:
    """Parse binary answer from text, similar to tasks.py"""
    match = re.search(r"\b(yes|no)\b", text.strip(), re.IGNORECASE)
    return match.group(1).capitalize() if match else text.strip()

def binary_vqa(image_url, prompt):
    messages = [
        {"role": "user", "content":[
            {"type": "image", "image":image_url},
            {"type": "text", "text":"Answer the following yes/no question about the image. "+prompt}
        ]}
    ]
    response = qwen_ft_caption(messages)
    return _parse_binary_answer(response)