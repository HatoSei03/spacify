import torch
from diffusers.utils import load_image
from img2panoimg import Image2360PanoramaImagePipeline
import sys
import time
from PIL import Image
from diffusers import FluxPipeline

# Nhận prompt từ command line
if len(sys.argv) > 1:
    prompt = ' '.join(sys.argv[1:])
else:
    prompt = 'A cozy living room with a comfortable sofa, a coffee table in the center, and a TV mounted on the wall.'

# Load mask
mask = load_image("./data/i2p-mask.jpg")

# Sinh ảnh 2D từ prompt bằng FLUX.1-dev
flux_model_id = "black-forest-labs/FLUX.1-dev"
flux_pipe = FluxPipeline.from_pretrained(flux_model_id, torch_dtype=torch.bfloat16)
flux_pipe.enable_model_cpu_offload()

start_flux = time.time()
flux_image = flux_pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
flux_image.save("flux2d.png")
end_flux = time.time()
print(f"FLUX.1-dev time: {end_flux - start_flux:.2f} seconds")

# for <16GB gpu
input = {'prompt': prompt, 'image': flux_image, 'mask': mask, 'upscale': False}
# for >16GB gpu (24GB at least)
# input = {'prompt': prompt, 'image': flux_image, 'mask': mask, 'upscale': True}

model_id = 'models'
img2panoimg = Image2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)
start_pano = time.time()
output = img2panoimg(input)
output.save('result.png')
end_pano = time.time()
print(f"Pano time: {end_pano - start_pano:.2f} seconds")