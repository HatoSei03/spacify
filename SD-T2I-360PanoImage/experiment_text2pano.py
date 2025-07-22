import os
import torch
import time
import csv
import numpy as np
from PIL import Image
from transformers import pipeline
from langdetect import detect
from diffusers.utils import load_image
from diffusers import FluxPipeline
from txt2panoimg import Text2360PanoramaImagePipeline
from img2panoimg import Image2360PanoramaImagePipeline
import gc
import clip
import open_clip

prompts = []
with open('./data/prompts.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('description'):
            prompts.append(row['description'].strip())

# Setup thư mục
os.makedirs('outputs/best_of_n', exist_ok=True)
os.makedirs('outputs/flux2d', exist_ok=True)
os.makedirs('outputs/flux2pano', exist_ok=True)

# CLIP setup: dùng MobileCLIP-S1 (clip-vit-base-patch16-224-multilingual-v2)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'clip-vit-base-patch16-224-multilingual-v2', pretrained='laion400m_e32', device=device
)
clip_model.eval()

def get_clip_score(image_path, prompt):
    with torch.no_grad():
        image = clip_preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
        text = open_clip.tokenize([prompt]).to(device)
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        similarity = (image_features @ text_features.T).item()
    del image, text, image_features, text_features
    torch.cuda.empty_cache()
    gc.collect()
    return similarity

def compute_mrapc(image_path):
    img = np.array(Image.open(image_path).convert('L'))
    diff_h = np.abs(img[:, 1:] - img[:, :-1])
    diff_v = np.abs(img[1:, :] - img[:-1, :])
    mrapc_score = - (diff_h.mean() + diff_v.mean()) / 2
    del img, diff_h, diff_v
    gc.collect()
    return mrapc_score

def compute_discontinuity_score(image_path):
    img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    left = img[:, 0, :]
    right = img[:, -1, :]
    score = np.mean(np.linalg.norm(left - right, axis=1))
    del img, left, right
    gc.collect()
    return score

# Model setup
model_id = 'models'
txt2panoimg = Text2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)
img2panoimg = Image2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)

flux_model_id = "black-forest-labs/FLUX.1-dev"

flux_pipe = DiffusionPipeline.from_pretrained(flux_model_id, torch_dtype=torch.bfloat16)
flux_pipe.to(device)
flux_pipe.vae.enable_slicing()
flux_pipe.vae.enable_tiling()
torch.backends.cuda.matmul.allow_tf32 = True

# Main experiment
N = 4  # Có thể giảm xuống 2 nếu vẫn OOM
results = []
for idx, prompt in enumerate(prompts):
    print(f"\n=== Prompt {idx+1}/{len(prompts)}: {prompt}")
    
    # Detect language and translate if needed (commented as in original)
    # try:
    #     lang = detect(prompt)
    # except Exception:
    #     lang = 'en'
    # if lang == 'vi':
    #     translator = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en", framework="pt")
    #     prompt_en = translator(prompt)[0]['translation_text']
    # else:
    prompt_en = prompt

    # Best-of-N: Sinh từng cái một, không giữ list lớn
    best_clip, best_mrapc, best_discont, best_path, best_time = -float('inf'), -float('inf'), float('inf'), None, 0
    t_best_start = time.time()
    pano1_path = None  # Để lưu cho hướng 1
    clip1, mrapc1, discont1, time1 = None, None, None, None
    for j in range(N):
        inputN = {'prompt': prompt_en, 'upscale': False}
        tN1 = time.time()
        with torch.no_grad():
            panoN = txt2panoimg(inputN)
        panoN_path = f'outputs/best_of_n/pano_{idx}_{j}.png'
        panoN.save(panoN_path)
        tN2 = time.time()
        clipN = get_clip_score(panoN_path, prompt_en)
        mrapcN = compute_mrapc(panoN_path)
        discontN = compute_discontinuity_score(panoN_path)
        
        # Lưu cho hướng 1 nếu là lần đầu
        if j == 0:
            pano1_path = panoN_path
            clip1 = clipN
            mrapc1 = mrapcN
            discont1 = discontN
            time1 = tN2 - tN1
        
        # Cập nhật best nếu tốt hơn (dùng tiêu chí của bạn)
        score_current = 0.5 * clipN + 0.5 * mrapcN
        score_best = 0.5 * best_clip + 0.5 * best_mrapc
        if score_current > score_best:
            best_clip, best_mrapc, best_discont, best_path, best_time = clipN, mrapcN, discontN, panoN_path, tN2 - tN1
        
        # Giải phóng ngay sau mỗi iteration
        del panoN
        torch.cuda.empty_cache()
        gc.collect()
    
    t_best_end = time.time()

    # FLUX2D + pano
    t3 = time.time()
    with torch.no_grad():
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
    with torch.no_grad():
        pano3 = img2panoimg(input3)
    pano3_path = f'outputs/flux2pano/pano_{idx}.png'
    pano3.save(pano3_path)
    t4 = time.time()
    clip3 = get_clip_score(pano3_path, prompt_en)
    mrapc3 = compute_mrapc(pano3_path)
    discont3 = compute_discontinuity_score(pano3_path)
    
    # Giải phóng
    del flux_image, mask, pano3
    torch.cuda.empty_cache()
    gc.collect()

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
        'time_best': t_best_end - t_best_start,
        'clip_flux': clip3,
        'mrapc_flux': mrapc3,
        'discont_flux': discont3,
        'time_flux': t4 - t3,
        'pano_one': pano1_path,
        'pano_best': best_path,
        'flux2d': flux2d_path,
        'pano_flux': pano3_path
    })
    print(f"  [one-shot]   CLIP={clip1:.4f} MRAPC={mrapc1:.4f} Discont={discont1:.4f} Time={time1:.2f}s")
    print(f"  [best-of-N]  CLIP={best_clip:.4f} MRAPC={best_mrapc:.4f} Discont={best_discont:.4f} Time={t_best_end - t_best_start:.2f}s")
    print(f"  [flux2pano]  CLIP={clip3:.4f} MRAPC={mrapc3:.4f} Discont={discont3:.4f} Time={t4 - t3:.2f}s")

# Xuất CSV
with open('outputs/experiment_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'idx', 'prompt',
        'clip_one', 'mrapc_one', 'discont_one', 'time_one',
        'clip_best', 'mrapc_best', 'discont_best', 'time_best',
        'clip_flux', 'mrapc_flux', 'discont_flux', 'time_flux',
        'pano_one', 'pano_best', 'flux2d', 'pano_flux'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print("\nExperiment completed. Results saved to outputs/experiment_results.csv")