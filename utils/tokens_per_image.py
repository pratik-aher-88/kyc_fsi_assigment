import requests
from PIL import Image
from transformers import AutoProcessor
import os

# Your image source - can be URL or local path
IMAGE_URL_OR_PATH = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"

def load_image(source):
    """Load image from URL or local file path"""
    if source.startswith(('http://', 'https://')):
        print(f"Downloading image from URL: {source}")
        response = requests.get(source)
        response.raise_for_status()
        return Image.open(requests.get(source, stream=True).raw)
    else:
        print(f"Loading image from path: {source}")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
        return Image.open(source)

def count_image_tokens(image):
    """Count how many tokens an image takes using Qwen 2.5 VL processor"""
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What's in this image?"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    
    # Count the image pad tokens (151655 is Qwen2.5 VL's image token ID)
    image_tokens = (input_ids == 151655).sum().item()
    
    return image_tokens, input_ids

def main():
    import sys
    
    image_source = sys.argv[1] if len(sys.argv) > 1 else IMAGE_URL_OR_PATH
    
    print(f"Processing image: {image_source}")
    image = load_image(image_source)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    print("\nCalculating tokens...")
    image_tokens, input_ids = count_image_tokens(image)
    
    print(f"Total tokens: {len(input_ids)}")
    print(f"Image tokens: {image_tokens}")
    print(f"Text tokens: {len(input_ids) - image_tokens}")
    
if __name__ == "__main__":
    main()