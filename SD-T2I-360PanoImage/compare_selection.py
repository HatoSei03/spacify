import csv
from PIL import Image
import matplotlib.pyplot as plt

# Đọc kết quả từ file csv
results = []
with open('panorama_selection_results.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row['clip'] = float(row['clip'])
        row['mrapc'] = float(row['mrapc'])
        row['time'] = float(row['time'])
        results.append(row)

# Phương pháp cũ: chọn ảnh đầu tiên (hoặc random)
old = results[0]
# Phương pháp mới: chọn ảnh tốt nhất theo weighted sum (đã chọn trong demo_t2p.py)
alpha = 0.5
new = max(results, key=lambda x: alpha*x['clip'] + (1-alpha)*x['mrapc'])

# Xuất bảng so sánh
print("\nBẢNG SO SÁNH PHƯƠNG PHÁP CHỌN ẢNH PANORAMA")
print(f"{'Phương pháp':<15} | {'Ảnh':<15} | {'CLIP':<8} | {'MRAPC':<8} | {'Time(s)':<8}")
print("-"*65)
print(f"{'Cũ':<15} | {old['img']:<15} | {old['clip']:<8.4f} | {old['mrapc']:<8.4f} | {old['time']:<8.2f}")
print(f"{'Mới':<15} | {new['img']:<15} | {new['clip']:<8.4f} | {new['mrapc']:<8.4f} | {new['time']:<8.2f}")

# Visualize hai ảnh
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].imshow(Image.open(old['img']))
axs[0].set_title(f"Cũ: {old['img']}\nCLIP={old['clip']:.4f}, MRAPC={old['mrapc']:.4f}")
axs[0].axis('off')
axs[1].imshow(Image.open(new['img']))
axs[1].set_title(f"Mới: {new['img']}\nCLIP={new['clip']:.4f}, MRAPC={new['mrapc']:.4f}")
axs[1].axis('off')
plt.tight_layout()
plt.savefig('compare_selection.png')
plt.show()

# Nhận xét tự động
print("\nNhận xét tự động:")
if new['clip'] > old['clip']:
    print("- Ảnh mới có CLIP cao hơn (semantic sát prompt hơn).")
else:
    print("- Ảnh mới có CLIP tương đương hoặc thấp hơn ảnh cũ.")
if new['mrapc'] > old['mrapc']:
    print("- Ảnh mới có MRAPC cao hơn (cấu trúc hình học tốt hơn, ít artifact hơn).")
else:
    print("- Ảnh mới có MRAPC tương đương hoặc thấp hơn ảnh cũ.")
print("- Thời gian sinh ảnh mới: %.2fs, ảnh cũ: %.2fs" % (new['time'], old['time'])) 