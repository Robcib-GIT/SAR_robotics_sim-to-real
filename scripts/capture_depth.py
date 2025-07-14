#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

class DepthImageSaver:
    def __init__(self):
        rospy.init_node("depth_image_saver", anonymous=True)
        self.bridge = CvBridge()
        self.image_count = 0

        # Carpeta donde guardar las imágenes
        self.output_dir = "/home/robcib/Desktop/Christyan/dataset_"
        os.makedirs(self.output_dir, exist_ok=True)

        # Suscribirse al tópico de profundidad
        rospy.Subscriber("/depth", Image, self.callback)

        print("[INFO] Esperando imágenes de profundidad en /depth...")
        rospy.spin()

    def callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            print(f"[INFO] Tamaño de la imagen recibida: {depth_image.shape}, dtype: {depth_image.dtype}")

            normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            normalized = np.uint8(normalized)
            print(f"[INFO] Tamaño normalizado: {normalized.shape}, dtype: {normalized.dtype}")

            file_path = os.path.join(self.output_dir, f"depth_{self.image_count:05d}.png")
            cv2.imwrite(file_path, normalized)
            
            img_check = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            print(f"[INFO] Imagen guardada tiene forma: {img_check.shape}")

            self.image_count += 1

        except Exception as e:
            print(f"[ERROR] Error procesando la imagen: {e}")

if __name__ == "__main__":
    DepthImageSaver()

