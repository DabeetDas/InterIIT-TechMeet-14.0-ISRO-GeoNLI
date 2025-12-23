from modal_request import qwen_base

def general_vqa(messages):
    response = qwen_base(messages)
    
    return response

