import torch
from transformers import pipeline  # Added for translation
from langdetect import detect  # Added for language detection; install with pip install langdetect
from txt2panoimg import Text2360PanoramaImagePipeline
import sys
import time
from PIL import Image
import numpy as np
import clip

# Lấy prompt từ command line nếu có, nếu không dùng mặc định
if len(sys.argv) > 1:
    prompt = ' '.join(sys.argv[1:])
else:
    prompt = 'The living room'
# Detect language and translate if Vietnamese
try:
    lang = detect(prompt)
except Exception:
    lang = 'en'  # Default to English if detection fails 

if lang == 'vi':
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en", framework="pt")  # Added framework="pt" to force PyTorch and avoid TF import errors
    translated_prompt = translator(prompt)[0]['translation_text']
    print(f"Original prompt (Vietnamese): {prompt}")
    print(f"Translated prompt (English): {translated_prompt}")
    prompt = translated_prompt

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
    # MRAPC: Multi-region Adjacent Pixels Correlation (no-reference)
    # Simple version: mean absolute diff of adjacent pixels (horizontal + vertical)
    img = np.array(Image.open(image_path).convert('L'))
    diff_h = np.abs(img[:, 1:] - img[:, :-1])
    diff_v = np.abs(img[1:, :] - img[:-1, :])
    # Lower mean diff = higher structural consistency
    mrapc_score = - (diff_h.mean() + diff_v.mean()) / 2
    return mrapc_score

# Sinh nhiều ảnh và chọn ảnh tốt nhất
N = 4  # Số lượng ảnh sinh ra cho mỗi prompt
model_id = 'models'
txt2panoimg = Text2360PanoramaImagePipeline(model_id, torch_dtype=torch.float16)

results = []
for i in range(N):
    # for <16GB gpu
    # input = {'prompt': prompt, 'upscale': False}
    # for >16GB gpu (24GB at least)
    # input = {'prompt': prompt, 'upscale': True}
    input = {'prompt': prompt, 'upscale': False}
    start = time.time()
    output = txt2panoimg(input)
    img_path = f'result_{i}.png'
    output.save(img_path)
    end = time.time()
    clip_score = get_clip_score(img_path, prompt)
    mrapc_score = compute_mrapc(img_path)
    results.append({'img': img_path, 'clip': clip_score, 'mrapc': mrapc_score, 'time': end-start})
    print(f"Image {img_path}: CLIP={clip_score:.4f}, MRAPC={mrapc_score:.4f}, Time={end-start:.2f}s")

# Chọn ảnh tốt nhất (weighted sum, alpha=0.5)
alpha = 0.5
best = max(results, key=lambda x: alpha*x['clip'] + (1-alpha)*x['mrapc'])
print(f"Best image: {best['img']} | CLIP={best['clip']:.4f} | MRAPC={best['mrapc']:.4f} | Time={best['time']:.2f}s")

# Lưu bảng kết quả cho so sánh
import csv
with open('panorama_selection_results.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['img', 'clip', 'mrapc', 'time'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)
