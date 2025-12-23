import requests

base_url = "https://justharshitjaiswal14--vlm-inference-server-fastapi-e-17eb43-dev.modal.run"
def llama3_8b(messages):

    # response = requests.post('https://shlokjain0177--vlm-inference-server-fastapi-entrypoint-dev.modal.run/llama3_3b', json={'messages': messages})
    import json
    response = requests.post(
        f'{base_url}/llama3_3b',
        params={'messages': json.dumps(messages)}
    )
    print(response)
    print(response.json())
    return response.json().get('response', '')

def qwen_base(messages):

    # response = requests.post('https://shlokjain0177--vlm-inference-server-fastapi-entrypoint-dev.modal.run/qwen_base', json={'messages': messages})
    import json
    response = requests.post(
        f'{base_url}/qwen_base',
        params={'messages': json.dumps(messages)}
    )
    print(response)
    print(response.json())
    return response.json().get('response', '')

def qwen_ft_caption(messages):
    print(messages)
    print(type(messages))
    import json
    # FastAPI endpoint expects messages as query parameter (based on error message)
    # Send as JSON-encoded string in query parameter
    response = requests.post(
        f'{base_url}/qwen_caption',
        params={'messages': json.dumps(messages)}
    )
    print(response)
    print(response.json())
    return response.json().get('response', '')

def qwen_ft_ground(messages):

    # response = requests.post('https://shlokjain0177--vlm-inference-server-fastapi-entrypoint-dev.modal.run/qwen_grounding', json={'messages': messages})
    import json
    response = requests.post(
        f'{base_url}/qwen_grounding',
        params={'messages': json.dumps(messages)}
    )
    print(response)
    print(response.json())
    return response.json()

def qwen_semantic_vqa(messages):

    # response = requests.post('https://shlokjain0177--vlm-inference-server-fastapi-entrypoint-dev.modal.run/qwen_semantic_vqa', json={'messages': messages})
    import json
    response = requests.post(
        f'{base_url}/qwen_semantic_vqa',
        params={'messages': json.dumps(messages)}
    )
    print(response)
    print(response.json())
    return response.json().get('response', '')

def llama3_8b_guide(messages):
    # return normal llama for now
    return llama3_8b(messages)

def sam_ensemble_mask(image_url: str, obb_box: list, text_query: str):
    """
    Call Modal endpoint for SAM ensemble mask generation.
    
    Args:
        image_url: URL of the image
        obb_box: Normalized OBB box [cx, cy, w, h, angle] where cx, cy, w, h are in [0, 1]
        text_query: Text query for RemoteCLIP scoring
    
    Returns:
        Dictionary with 'mask' (base64 encoded), 'mask_shape', 'best_model', 'best_score'
    """
    import json
    import requests
    
    payload = {
        'image_url': image_url,
        'obb_box': obb_box,
        'text_query': text_query
    }
    
    response = requests.post(
        f'{base_url}/sam_ensemble_mask',
        params={'payload': json.dumps(payload)}
    )
    print(response)
    print(response.json())
    return response.json()