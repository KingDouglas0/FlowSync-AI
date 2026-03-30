import patches
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

frame = cv2.imread(r"C:\FlowSync AI\test_media\test7.jpg")

if frame is None:
    print("❌ Image NOT found. Check path.")
    exit()
else:
    print("✅ Image loaded successfully!")

results = model(frame)
results[0].show()