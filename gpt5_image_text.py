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
                # 1. UPDATED MODEL LIST
                "model": ([
                    "gpt-5.2", 
                    "gpt-5.2-pro", 
                    "gpt-5", 
                    "gpt-5-mini", 
                    "gpt-5-nano", 
                    "gpt-4.1", 
                    "gpt-4o", 
                    "gpt-4o-mini", 
                    "gpt-4-turbo",
                    "o1-preview", 
                    "o1-mini"
                ],),
                "openai_key": ("STRING", {"default": "your_openai_key_here", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                # Standard limit for output tokens
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 32768, "step": 1}),
                # 2. NEW: Explicit input for reasoning models (o1 / gpt-5)
                "max_completion_tokens": ("INT", {"default": 10000, "min": 1, "max": 65536, "step": 1}),
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

    def analyze(self, prompt, system_prompt, model, openai_key, temperature, max_tokens, max_completion_tokens, image=None):
        if openai_key == "your_openai_key_here":
            raise ValueError("Please set your OpenAI API key in the node.")
        
        client = openai.OpenAI(api_key=openai_key)
        user_content = [{"type": "text", "text": prompt}]
        
        # Handle Image Input
        if image is not None:
            batch_size = image.shape[0] if len(image.shape) == 4 else 1
            for b in range(min(batch_size, 10)):
                single_image = image[b:b + 1] if batch_size > 1 else image
                pil_image = tensor2pil(single_image)
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })
        
        # 3. Detect Model Type
        # GPT-5 and o1 are "reasoning" models
        is_reasoning_model = model.startswith("gpt-5") or model.startswith("o1-")

        api_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        # 4. Apply Correct Token Parameter
        if is_reasoning_model:
            # Reasoning models use 'max_completion_tokens' and NO temperature
            api_kwargs["max_completion_tokens"] = max_completion_tokens
            # Do NOT add temperature to api_kwargs for reasoning models
        else:
            # Standard models use 'max_tokens' and 'temperature'
            api_kwargs["max_tokens"] = max_tokens
            api_kwargs["temperature"] = temperature

        try:
            response = client.chat.completions.create(**api_kwargs)
            choice = response.choices[0]
            
            # Check for length finish reason
            if choice.finish_reason == "length" and not choice.message.content:
                limit_used = max_completion_tokens if is_reasoning_model else max_tokens
                raise ValueError(
                    f"Model ran out of tokens! Limit was {limit_used}. Increase 'max_completion_tokens' (if reasoning) or 'max_tokens'."
                )

            if not choice.message.content:
                raise ValueError("No content in response.")
            
            return (choice.message.content.strip(),)
            
        except openai.OpenAIError as e:
            raise ValueError(f"OpenAI API error: {str(e)}")
