import os
import glob
import subprocess
import time

# Đường dẫn tới folder ảnh input
input_dir = './data/selected_bestofn_images'
# Đường dẫn tới folder tổng output
output_root = './Pro_inpaint_50'
os.makedirs(output_root, exist_ok=True)

# Lấy danh sách tất cả ảnh panorama
image_paths = glob.glob(os.path.join(input_dir, '*.png'))
image_paths += glob.glob(os.path.join(input_dir, '*.jpg'))
image_paths.sort() 

start_time = time.time()

for img_path in image_paths:
    # Tên file không đuôi
    base = os.path.splitext(os.path.basename(img_path))[0]
    # Folder con cho từng panorama
    out_dir = os.path.join(output_root, base)
    os.makedirs(out_dir, exist_ok=True)
    # Gọi pro_inpaint.py, truyền đường dẫn ảnh và output_dir qua biến môi trường
    # Mọi output sẽ được lưu vào out_dir (folder con riêng cho từng ảnh)
    env = os.environ.copy()
    env['PRO_INPAINT_INPUT'] = img_path
    env['PRO_INPAINT_OUTDIR'] = out_dir
    # Chạy không lưu output trung gian (output sẽ nằm trong out_dir)
    subprocess.run(['python', 'pro_inpaint.py'], env=env)
    # Kiểm tra file output chính
    pano_final = os.path.join(out_dir, 'panorama_final.jpg')
    if not os.path.exists(pano_final):
        print(f'WARNING: Không tìm thấy {pano_final} sau khi xử lý {img_path}') 

end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / len(image_paths) if image_paths else 0
print(f"Tổng thời gian: {total_time:.2f} giây cho {len(image_paths)} ảnh")
print(f"Thời gian trung bình mỗi ảnh: {avg_time:.2f} giây") 