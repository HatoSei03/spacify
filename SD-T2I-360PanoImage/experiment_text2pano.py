import os
import torch
import time
import csv
import numpy as np
from PIL import Image
from transformers import pipeline
from langdetect import detect
import clip
from diffusers.utils import load_image
from diffusers import FluxPipeline
from txt2panoimg import Text2360PanoramaImagePipeline
from img2panoimg import Image2360PanoramaImagePipeline

prompts = []
with open('prompts.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('description'):
            prompts.append(row['description'].strip())
# Setup
os.makedirs('outputs/best_of_n', exist_ok=True)
os.makedirs('outputs/flux2d', exist_ok=True)
os.makedirs('outputs/flux2pano', exist_ok=True)

# CLIP setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def get_clip_score(image_path, prompt):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        similarity = (image_features @ text_features.T).item()
    return similarity

def compute_mrapc(image_path):
    img = np.array(Image.open(image_path).convert('L'))
    diff_h = np.abs(img[:, 1:] - img[:, :-1])
    diff_v = np.abs(img[1:, :] - img[:-1, :])
    mrapc_score = - (diff_h.mean() + diff_v.mean()) / 2
    return mrapc_score

def compute_discontinuity_score(image_path):
    img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    left = img[:, 0, :]
    right = img[:, -1, :]
    score = np.mean(np.linalg.norm(left - right, axis=1))
    return score

# Model setup
model_id = 'models'
txt2panoimg = Text2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)
img2panoimg = Image2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)
flux_model_id = "black-forest-labs/FLUX.1-dev"
flux_pipe = FluxPipeline.from_pretrained(flux_model_id, torch_dtype=torch.bfloat16)
flux_pipe.enable_model_cpu_offload()

# Main experiment
N = 4  # Số lượng ảnh cho best-of-N
results = []
for idx, prompt in enumerate(prompts):
    print(f"\n=== Prompt {idx+1}/{len(prompts)}: {prompt}")
    # Detect language and translate if needed
    try:
        lang = detect(prompt)
    except Exception:
        lang = 'en'
    if lang == 'vi':
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en", framework="pt")
        prompt_en = translator(prompt)[0]['translation_text']
    else:
        prompt_en = prompt

    # 2. Best-of-N (hướng 2) - sinh N ảnh, lấy ảnh đầu tiên cho hướng 1
    best_clip, best_mrapc, best_discont, best_path, best_time = None, None, None, None, None
    t_best_start = time.time()
    panoN_paths = []
    clipN_list, mrapcN_list, discontN_list, timeN_list = [], [], [], []
    for j in range(N):
        inputN = {'prompt': prompt_en, 'upscale': False}
        tN1 = time.time()
        panoN = txt2panoimg(inputN)
        panoN_path = f'outputs/best_of_n/pano_{idx}_{j}.png'
        panoN.save(panoN_path)
        tN2 = time.time()
        clipN = get_clip_score(panoN_path, prompt_en)
        mrapcN = compute_mrapc(panoN_path)
        discontN = compute_discontinuity_score(panoN_path)
        panoN_paths.append(panoN_path)
        clipN_list.append(clipN)
        mrapcN_list.append(mrapcN)
        discontN_list.append(discontN)
        timeN_list.append(tN2-tN1)
        if best_clip is None or (0.5*clipN + 0.5*mrapcN) > (0.5*best_clip + 0.5*best_mrapc):
            best_clip, best_mrapc, best_discont, best_path, best_time = clipN, mrapcN, discontN, panoN_path, tN2-tN1
    t_best_end = time.time()

    # Hướng 1: lấy ảnh đầu tiên của best-of-N
    pano1_path = panoN_paths[0]
    clip1 = clipN_list[0]
    mrapc1 = mrapcN_list[0]
    discont1 = discontN_list[0]
    time1 = timeN_list[0]

    # 3. FLUX2D + pano (hướng 3)
    t3 = time.time()
    flux_image = flux_pipe(
        prompt_en,
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    flux2d_path = f'outputs/flux2d/flux2d_{idx}.png'
    flux_image.save(flux2d_path)
    mask = load_image("./data/i2p-mask.jpg")
    input3 = {'prompt': prompt_en, 'image': flux_image, 'mask': mask, 'upscale': False}
    pano3 = img2panoimg(input3)
    pano3_path = f'outputs/flux2pano/pano_{idx}.png'
    pano3.save(pano3_path)
    t4 = time.time()
    clip3 = get_clip_score(pano3_path, prompt_en)
    mrapc3 = compute_mrapc(pano3_path)
    discont3 = compute_discontinuity_score(pano3_path)

    # Ghi kết quả
    results.append({
        'idx': idx,
        'prompt': prompt,
        'clip_one': clip1,
        'mrapc_one': mrapc1,
        'discont_one': discont1,
        'time_one': time1,
        'clip_best': best_clip,
        'mrapc_best': best_mrapc,
        'discont_best': best_discont,
        'time_best': t_best_end-t_best_start,
        'clip_flux': clip3,
        'mrapc_flux': mrapc3,
        'discont_flux': discont3,
        'time_flux': t4-t3,
        'pano_one': pano1_path,
        'pano_best': best_path,
        'flux2d': flux2d_path,
        'pano_flux': pano3_path
    })
    print(f"  [one-shot]   CLIP={clip1:.4f} MRAPC={mrapc1:.4f} Discont={discont1:.4f} Time={time1:.2f}s")
    print(f"  [best-of-N]  CLIP={best_clip:.4f} MRAPC={best_mrapc:.4f} Discont={best_discont:.4f} Time={t_best_end-t_best_start:.2f}s")
    print(f"  [flux2pano]  CLIP={clip3:.4f} MRAPC={mrapc3:.4f} Discont={discont3:.4f} Time={t4-t3:.2f}s")

# Xuất file csv tổng hợp
with open('outputs/experiment_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'idx','prompt',
        'clip_one','mrapc_one','discont_one','time_one',
        'clip_best','mrapc_best','discont_best','time_best',
        'clip_flux','mrapc_flux','discont_flux','time_flux',
        'pano_one','pano_best','flux2d','pano_flux'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("\nExperiment completed. Results saved to outputs/experiment_results.csv") 