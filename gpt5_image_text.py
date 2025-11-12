import torch
import openai
import base64
import numpy as np
from io import BytesIO
from PIL import Image

def tensor2pil(image_tensor):
    i = 255. * image_tensor[0].cpu().numpy()
    image = np.clip(i, 0, 255).astype(np.uint8)
    c = image.shape[-1]
    if c == 1:
        image = np.repeat(image, 3, axis=-1)
    elif c == 3:
        pass
    elif c == 4:
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported channels: {c}.")
    return Image.fromarray(image, mode='RGB')

class GPT5ImageText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Analyze this image and text.", "multiline": True}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "model": (["gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-preview"],),  # Updated for 2025 models
                "openai_key": ("STRING", {"default": "your_openai_key_here"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),  # Increased max for GPT-5
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "analyze"
    CATEGORY = "openai/analysis"
    OUTPUT_NODE = True

    def analyze(self, prompt, system_prompt, model, openai_key, temperature, max_tokens, image=None):
        if openai_key == "your_openai_key_here":
            raise ValueError("Please set your OpenAI API key in the node.")
        
        # Dynamically select token param based on model (fixes the error)
        token_param = "max_completion_tokens" if model == "gpt-5" or model.startswith("o1-") else "max_tokens"
        
        client = openai.OpenAI(api_key=openai_key)
        user_content = [{"type": "text", "text": prompt}]
        
        if image is not None:
            batch_size = image.shape[0] if len(image.shape) == 4 else 1
            for b in range(min(batch_size, 10)):  # OpenAI limit: ~10 images
                single_image = image[b:b + 1] if batch_size > 1 else image
                pil_image = tensor2pil(single_image)
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                **{token_param: max_tokens},  # Dynamic param injection
            )
            
            if not response.choices or response.choices[0].message.content is None:
                raise ValueError("No content in response")
            
            return (response.choices[0].message.content.strip(),)
            
        except openai.OpenAIError as e:
            raise ValueError(f"OpenAI API error: {str(e)}. Check model/token limits.")
