from ultralytics import YOLO
import cv2
import math

model_path = r"program\model\gabungNewV8version4-200x320-26juni2025-x3.pt"
video_path = r"program\vid\video_short_basket_yusuf.mp4"

model = YOLO(model_path)
classNames = model.names  

try:

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise FileNotFoundError("Video path tidak ditemukan atau format tidak didukung!")
    
    print("Video berhasil dimuat, loading detection...")
    
    frame_skip = 2  
    frame_count = 0
    
    resize_factor = 0.7  
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Video selesai atau tidak dapat membaca frame.")
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        if resize_factor != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        
        results = model(frame, imgsz=320) 
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if resize_factor != 1.0:
                    x1, y1, x2, y2 = int(x1/resize_factor), int(y1/resize_factor), int(x2/resize_factor), int(y2/resize_factor)

                    display_frame = cv2.resize(frame, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
                else:
                    display_frame = frame
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (51, 85, 170), 2)
                label = f"{classNames[cls]} {confidence:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if classNames[cls] == "side":
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.line(display_frame, (x1, center_y), (x2, center_y), (0, 255, 0), 2)
                    cv2.line(display_frame, (center_x, y1), (center_x, y2), (0, 255, 0), 2)
                    cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1) 

        cv2.imshow("Object Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()