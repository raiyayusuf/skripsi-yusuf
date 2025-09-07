import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load model YOLO
model_path = "program/model/gabungNewV8version4-200x320-26juni2025-x3.pt"
model = YOLO(model_path)

# Load gambar
image_path = "program/img/lapangan1.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Lakukan deteksi objek
results = model(image)

# Visualisasi hasil deteksi
result_image = results[0].plot()

# Tampilkan hasil
plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.axis('off')
plt.title('Hasil Object Detection')
plt.show()

# Simpan hasil jika diperlukan
output_path = "program/img/lapangan1_detection.jpg"
cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
print(f"Hasil deteksi disimpan di: {output_path}")

# Tampilkan informasi deteksi
for i, det in enumerate(results[0].boxes):
    class_id = int(det.cls)
    confidence = float(det.conf)
    class_name = model.names[class_id]
    print(f"Deteksi {i+1}: {class_name} (confidence: {confidence:.2f})")