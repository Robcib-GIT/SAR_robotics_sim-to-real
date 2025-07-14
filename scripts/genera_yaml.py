import os
from collections import Counter

# Ruta al dataset
dataset_path = "/home/robcib/Desktop/Christyan/dataset unido/dataset_cnn/dataset"
labels_dirs = [
    os.path.join(dataset_path, split, "labels")
    for split in ["train", "val", "test"]
]

# Contar class_ids en los archivos .txt
class_counter = Counter()

for labels_dir in labels_dirs:
    if not os.path.exists(labels_dir):
        continue
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, fname), "r") as f:
            for line in f:
                class_id = line.strip().split()[0]
                class_counter[int(class_id)] += 1

# Construir lista de clases
sorted_classes = sorted(class_counter.items())
num_classes = len(sorted_classes)
class_names = [f"class{cls_id}" for cls_id, _ in sorted_classes]

# Crear contenido de data.yaml
yaml_content = f"""path: {dataset_path}
train: train/images
val: val/images
test: test/images

nc: {num_classes}
names: {class_names}
"""

# Guardar como data.yaml
with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("✅ Archivo data.yaml generado con éxito:")
print(yaml_content)
