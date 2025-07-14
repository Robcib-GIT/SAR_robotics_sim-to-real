import os
import json
import numpy as np
from PIL import Image
import cv2
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

input_dir = "/home/robcib/Desktop/Christyan/dataset_2_poor_ligth/semantic/"
json_dir = "/home/robcib/Desktop/Christyan/dataset_2_poor_ligth/json/"
output_yolo_seg_dir = "labels_seg"
output_voc_seg_dir = "labels_voc_seg"

os.makedirs(output_yolo_seg_dir, exist_ok=True)
os.makedirs(output_voc_seg_dir, exist_ok=True)

class_to_id = {}
class_id_counter = 0

def parse_rgba(rgba_str):
    return tuple(map(int, rgba_str.strip("()").split(", ")))

for fname in os.listdir(input_dir):
    if not fname.endswith(".png"):
        continue

    base_num = fname.split("_")[-1].split(".")[0]
    json_name = f"semantic_segmentation_labels_{base_num}.json"
    json_path = os.path.join(json_dir, json_name)

    if not os.path.exists(json_path):
        print(f"⚠️ No existe JSON para {fname} ({json_name}), saltando.")
        continue

    with open(json_path, "r") as f:
        color_class_map = json.load(f)

    image_path = os.path.join(input_dir, fname)
    image = Image.open(image_path).convert("RGBA")
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    yolo_seg_lines = []
    voc_objects = []

    for color_str, class_data in color_class_map.items():
        class_name = class_data["class"]

        # ❌ Excluir clases no deseadas
        if class_name.upper() in ["BACKGROUND", "UNLABELLED","GROUND"]:
            continue

        if class_name not in class_to_id:
            class_to_id[class_name] = class_id_counter
            class_id_counter += 1
        class_id = class_to_id[class_name]

        rgba = parse_rgba(color_str)
        mask = np.all(img_np == rgba, axis=-1).astype(np.uint8)

        if np.sum(mask) == 0:
            continue

        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 3:
                continue

            norm_points = []
            for point in cnt.squeeze():
                x, y = point
                norm_x = x / w
                norm_y = y / h
                norm_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
            yolo_line = f"{class_id} " + " ".join(norm_points)
            yolo_seg_lines.append(yolo_line)

            polygon_points = [{"x": int(p[0][0]), "y": int(p[0][1])} for p in cnt]
            voc_objects.append({
                "name": class_name,
                "polygon": polygon_points
            })

    base_name = os.path.splitext(fname)[0]
    yolo_path = os.path.join(output_yolo_seg_dir, base_name + ".txt")
    with open(yolo_path, "w") as f:
        f.write("\n".join(yolo_seg_lines))

    annotation = Element("annotation")
    SubElement(annotation, "filename").text = fname
    size = SubElement(annotation, "size")
    SubElement(size, "width").text = str(w)
    SubElement(size, "height").text = str(h)
    SubElement(size, "depth").text = "4"

    for obj in voc_objects:
        obj_elem = SubElement(annotation, "object")
        SubElement(obj_elem, "name").text = obj["name"]
        polygon = SubElement(obj_elem, "polygon")
        for pt in obj["polygon"]:
            pt_elem = SubElement(polygon, "pt")
            SubElement(pt_elem, "x").text = str(pt["x"])
            SubElement(pt_elem, "y").text = str(pt["y"])

    xml_str = parseString(tostring(annotation)).toprettyxml(indent="  ")
    xml_path = os.path.join(output_voc_seg_dir, base_name + ".xml")
    with open(xml_path, "w") as f:
        f.write(xml_str)

print("[✅] Conversión completa a YOLOv8-seg y PascalVOC extendido (con polígonos).")
print("Clases detectadas:")
for c, i in class_to_id.items():
    print(f"{i}: {c}")


# DATASET BBOX
"""import os
import json
import numpy as np
from PIL import Image
import cv2
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

input_dir = "/home/robcib/Desktop/Christyan/dataset_2_poor_ligth/semantic/"
json_dir = "/home/robcib/Desktop/Christyan/dataset_2_poor_ligth/json/"
output_yolo_dir = "labels_yolo"
output_voc_dir = "labels_voc"
os.makedirs(output_yolo_dir, exist_ok=True)
os.makedirs(output_voc_dir, exist_ok=True)

class_to_id = {}
class_id_counter = 0

def parse_rgba(rgba_str):
    return tuple(map(int, rgba_str.strip("()").split(", ")))

for fname in os.listdir(input_dir):
    if not fname.endswith(".png"):
        continue

    # Ejemplo: semantic_segmentation_0000.png --> semantic_segmentation_labels_0000.json
    base_num = fname.split("_")[-1].split(".")[0]  # "0000"
    json_name = f"semantic_segmentation_labels_{base_num}.json"
    json_path = os.path.join(json_dir, json_name)

    if not os.path.exists(json_path):
        print(f"¡No existe JSON para {fname} ({json_name}), saltando!")
        continue

    with open(json_path, "r") as f:
        color_class_map = json.load(f)

    image_path = os.path.join(input_dir, fname)
    image = Image.open(image_path).convert("RGBA")
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    yolo_labels = []
    voc_objects = []

    for color_str, class_data in color_class_map.items():
        class_name = class_data["class"]

        if class_name not in class_to_id:
            class_to_id[class_name] = class_id_counter
            class_id_counter += 1
        class_id = class_to_id[class_name]

        rgba = parse_rgba(color_str)
        mask = np.all(img_np == rgba, axis=-1).astype(np.uint8)

        if np.sum(mask) == 0:
            continue

        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h

            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            voc_objects.append({
                "name": class_name,
                "xmin": x,
                "ymin": y,
                "xmax": x + bw,
                "ymax": y + bh,
            })

    base_name = os.path.splitext(fname)[0]
    with open(os.path.join(output_yolo_dir, base_name + ".txt"), "w") as f:
        f.write("\n".join(yolo_labels))

    annotation = Element("annotation")
    SubElement(annotation, "filename").text = fname
    size = SubElement(annotation, "size")
    SubElement(size, "width").text = str(w)
    SubElement(size, "height").text = str(h)
    SubElement(size, "depth").text = "3"

    for obj in voc_objects:
        object_elem = SubElement(annotation, "object")
        SubElement(object_elem, "name").text = obj["name"]
        bndbox = SubElement(object_elem, "bndbox")
        SubElement(bndbox, "xmin").text = str(obj["xmin"])
        SubElement(bndbox, "ymin").text = str(obj["ymin"])
        SubElement(bndbox, "xmax").text = str(obj["xmax"])
        SubElement(bndbox, "ymax").text = str(obj["ymax"])

    xml_str = parseString(tostring(annotation)).toprettyxml(indent="  ")
    with open(os.path.join(output_voc_dir, base_name + ".xml"), "w") as f:
        f.write(xml_str)

print("[✅] Conversión completada.")
print("Clases encontradas:")
for c, i in class_to_id.items():
    print(f"{i}: {c}")"""



