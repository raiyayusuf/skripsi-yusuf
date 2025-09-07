from ultralytics import YOLO
import cv2
import math

model_path = r"program\model\gabungNewV8version4-200x320-26juni2025-x3.pt"
model = YOLO(model_path)
classNames = model.names  

image = r"program\img\lapangan2.jpg"

try:
    img = cv2.imread(image)
    if img is None:
        raise FileNotFoundError("Path tidak ditemukan!")
    print("Gambar berhasil dimuat, loading detection...")

    results = model(img)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (51, 85, 170), 2)
            label = f"{classNames[cls]} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if classNames[cls] == "side":
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.line(img, (x1, center_y), (x2, center_y), (0, 255, 0), 2)
                cv2.line(img, (center_x, y1), (center_x, y2), (0, 255, 0), 2)
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1) 

    cv2.imshow("Object Detection", img)
    cv2.waitKey(0)

except Exception as e:
    print(f"Error: {e}")

finally:
    cv2.destroyAllWindows()