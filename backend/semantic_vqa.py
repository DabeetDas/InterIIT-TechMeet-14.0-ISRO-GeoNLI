from modal_request import qwen_semantic_vqa
def semantic_vqa(image_url, prompt):
    messages = [
        {"role": "user", "content":[
            {"type": "image", "image":image_url},
            {"type": "text", "text":"Answer the following question within one to five words concisely. "+prompt}
        ]}
    ]
    response = qwen_semantic_vqa(messages)
    print(f"Response semantic: {response}")
    
    return response