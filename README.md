# ComfyUI-GPT5_Image_Text

A powerful **ComfyUI custom node** for **vision + text analysis** using **GPT-5 (and GPT-4o)** with **direct API key input**, **system prompt**, **temperature**, **max tokens**, and **multi-image support**.

Perfect for:
- Image captioning
- Visual question answering (VQA)
- Scene understanding
- Object detection via description
- Multimodal reasoning

No image generation or editing — **pure analysis only**.

<img width="958" height="452" alt="GPT5 Image & Text Node" src="https://github.com/user-attachments/assets/placeholder-gpt5-node-ui.png" />

> *(Replace with your own screenshot once the node is in ComfyUI)*

---

## Features

- **GPT-5 Ready** (when available) + GPT-4o / GPT-4o-mini fallbacks  
- **Multi-Image Input** – Analyze batches of images in one prompt  
- **Vision + Text** – Combine image(s) with text prompt  
- **Full Control** – System prompt, temperature, max tokens  
- **API Key in UI** – No `.env` files or hardcoding  
- **Zero Dependencies Beyond OpenAI** – Lightweight & fast  

---

## Installation

1. Clone this repo into `ComfyUI/custom_nodes/`:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-GPT5_Image_Text.git
