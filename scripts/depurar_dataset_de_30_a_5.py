import os
import shutil
import re

src_dir = "/home/robcib/Desktop/Christyan/dataset_"
dst_dir = "/home/robcib/Desktop/Christyan/dataset_1fps"
os.makedirs(dst_dir, exist_ok=True)

pattern = re.compile(r"(.*)_(\d+)\.(png|json)")

for fname in sorted(os.listdir(src_dir)):
    match = pattern.match(fname)
    if not match:
        continue

    prefix, index_str, ext = match.groups()
    index = int(index_str)

    if index % 5 == 0:  # Aproximadamente 1 imagen por segundo (asumiendo 30 FPS)
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        shutil.copy(src_path, dst_path)
        print(f"Copiado: {fname}")
