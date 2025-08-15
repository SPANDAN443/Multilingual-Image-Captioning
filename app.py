import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Avoid symlink warning on Windows

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from transformers import MarianMTModel, MarianTokenizer, pipeline, BartForConditionalGeneration, BartTokenizer, BlipProcessor, BlipForConditionalGeneration
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load translation model (multi-language to English)
print("Loading translation model...")
translation_model_name = "Helsinki-NLP/opus-mt-mul-en"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name).to(device)
translator = pipeline("translation", model=translation_model, tokenizer=translation_tokenizer)

# Load BART (creative content generation)
print("Loading BART model...")
content_model_name = "facebook/bart-large-cnn"
content_tokenizer = BartTokenizer.from_pretrained(content_model_name)
content_model = BartForConditionalGeneration.from_pretrained(content_model_name).to(device)

# Load BLIP for image captioning
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load Stable Diffusion with float16 for speed
print("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")

# Optional: Warm-up image to avoid first-call slowness
print("Warming up image generation...")
_ = pipe("A sunny beach with palm trees", guidance_scale=7.5, num_inference_steps=10).images[0]
print("Warm-up done.")

# Optional: Load ResNet (unused)
resnet_model_name = "microsoft/resnet-50"
feature_extractor = AutoFeatureExtractor.from_pretrained(resnet_model_name)
classification_model = AutoModelForImageClassification.from_pretrained(resnet_model_name)

# Image captioning function
def caption_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs, max_length=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Creative description generation
def generate_creative_content(caption):
    prompt = f"Write a creative description in 2-3 sentences about the scene in '{caption}', focusing on vivid imagery, colors, and emotions it evokes."
    inputs = content_tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True).to(device)
    summary_ids = content_model.generate(**inputs, max_length=100, min_length=30, num_beams=5, early_stopping=True)
    return content_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Main pipeline: translate → image → caption → creative content
def translate_and_generate_content(text):
    translation = translator(text, max_length=40)
    translated_text = translation[0]['translation_text']
    
    image = pipe(translated_text, guidance_scale=7.5, num_inference_steps=25).images[0]
    caption = caption_image(image)
    creative_content = generate_creative_content(caption)

    return translated_text, image, creative_content

# Gradio UI
interface = gr.Interface(
    fn=translate_and_generate_content,
    inputs="text",
    outputs=["text", "image", "text"],
    title="Multilingual to English AI Generator (Fast)"
)

# Launch
interface.launch(debug=True)
