import os
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoProcessor, SiglipVisionModel

from src.pipeline_flux_ipa import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.attention_processor import IPAFluxAttnProcessor2_0
from src.adapter import IPAdapter, resize_img

image_encoder_path = "google/siglip-so400m-patch14-384"
ipadapter_path = "models/strategy_adapter.bin"
    
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
)

ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)

image_dir = "./assets/strategy.png"
image_name = image_dir.split("/")[-1]
image = Image.open(image_dir).convert("RGB")
image = resize_img(image)

prompt = "3*3 puzzle of 9 sub-images, step-by-step painting process"
    
images = ip_model.generate(
    pil_image=image, 
    prompt=prompt,
    scale=0.9,
    width=1024, height=1024,
    seed=42
)

images[0].save(f"results/{image_name}")
