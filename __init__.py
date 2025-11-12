from .gpt5_image_text import GPT5ImageText

NODE_CLASS_MAPPINGS = {
    "GPT5ImageText": GPT5ImageText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT5ImageText": "GPT5 Image & Text"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']