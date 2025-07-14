import os
import shutil
import re

# Rutas
base_dir = "/home/robcib/Desktop/Christyan/dataset_2_poor_ligth/dataset_yolo"
output_dir = "/home/robcib/Desktop/Christyan/dataset_2_poor_ligth/dataset_prefixed"

# Crear salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Obtener archivos
images = sorted([f for f in os.listdir(base_dir) if f.startswith("rgb_") and f.endswith(".png")], key=lambda x: int(re.search(r'\d+', x).group()))
annotations = sorted([f for f in os.listdir(base_dir) if f.startswith("semantic_segmentation_") and f.endswith(".txt")], key=lambda x: int(re.search(r'\d+', x).group()))

# Validación mínima
if len(images) != len(annotations):
    print(f"⚠️ Hay {len(images)} imágenes y {len(annotations)} etiquetas. Usando mínimo común.")
    min_len = min(len(images), len(annotations))
    images = images[:min_len]
    annotations = annotations[:min_len]

# Procesar archivos
for img_name, ann_name in zip(images, annotations):
    img_src = os.path.join(base_dir, img_name)
    ann_src = os.path.join(base_dir, ann_name)

    # Nuevo nombre con prefijo
    new_img_name = "d2_" + img_name
    base_number = re.search(r'\d+', img_name).group()
    new_ann_name = f"d2_rgb_{base_number}.txt"

    img_dst = os.path.join(output_dir, new_img_name)
    ann_dst = os.path.join(output_dir, new_ann_name)

    shutil.copy(img_src, img_dst)
    shutil.copy(ann_src, ann_dst)

print("✅ Imágenes y etiquetas copiadas y renombradas correctamente.")

