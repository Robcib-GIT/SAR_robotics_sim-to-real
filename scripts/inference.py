import os
from ultralytics import YOLO
import cv2

# Rutas
images_dir = "/home/robcib/Desktop/Christyan/Nave_texturas"
model_path = "/home/robcib/Desktop/Christyan/runs_segment/yolov8s_seg_d12/weights/best.pt"

# Crear carpeta de salida paralela
output_dir = os.path.join(os.path.dirname(images_dir), "Nave_texturas_pred")
os.makedirs(output_dir, exist_ok=True)

# Cargar modelo YOLOv8-seg entrenado
model = YOLO(model_path)

# Obtener imágenes PNG
image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]

print(f"🔍 Procesando {len(image_files)} imágenes...")

# Inferencia sobre cada imagen
for img_name in image_files:
    img_path = os.path.join(images_dir, img_name)
    result = model.predict(source=img_path, save=False, save_txt=False, conf=0.25)

    # Dibujar resultados sobre la imagen original
    rendered = result[0].plot()

    # Guardar en carpeta de salida
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, rendered)
    print(f"✅ Guardado: {output_path}")

print(f"\n🎉 Inferencia completada. Resultados en: {output_dir}")
