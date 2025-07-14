
import os
import shutil
import random

# Ruta de entrada
input_dir = "/home/robcib/Desktop/Christyan/dataset unido/dataset_pascal"

# Ruta de salida base
output_base = "/home/robcib/Desktop/Christyan/dataset unido/dataset_pascal/split"

# Porcentaje de división
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Crear carpetas de salida
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)

# Obtener todas las imágenes y asegurarse de que tengan su .txt correspondiente
image_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
paired_files = [(f, f.replace(".png", ".xml")) for f in image_files if os.path.exists(os.path.join(input_dir, f.replace(".png", ".xml")))]

# Mezclar aleatoriamente
random.seed(42)
random.shuffle(paired_files)

# Dividir
total = len(paired_files)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_set = paired_files[:train_end]
val_set = paired_files[train_end:val_end]
test_set = paired_files[val_end:]

split_data = {'train': train_set, 'val': val_set, 'test': test_set}

# Copiar archivos
for split, items in split_data.items():
    for img_name, label_name in items:
        shutil.copy(os.path.join(input_dir, img_name), os.path.join(output_base, split, 'images', img_name))
        shutil.copy(os.path.join(input_dir, label_name), os.path.join(output_base, split, 'labels', label_name))

print("✅ División completada:")
print(f"  Train: {len(train_set)} muestras")
print(f"  Val:   {len(val_set)} muestras")
print(f"  Test:  {len(test_set)} muestras")

