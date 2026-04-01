from ultralytics import YOLO
import cv2
from flask import Flask, jsonify
import threading
import time

# Load model
model = YOLO('yolov8n.pt')

# Flask setup
app = Flask(__name__)

# Shared data
latest_data = {
    "lane_A": 0,
    "lane_B": 0,
    "lane_C": 0,
    "green": "A"
}

@app.route('/traffic', methods=['GET'])
def get_traffic():
    return jsonify(latest_data)

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

threading.Thread(target=run_server, daemon=True).start()

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

def count_vehicles(frame):
    results = model(frame)
    boxes = results[0].boxes

    count = 0
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                count += 1

    return count


while True:
    # === LOAD IMAGES (simulate live traffic) ===
    img_A = cv2.imread(r"C:\FlowSync AI\test_media\test7.jpg")
    img_B = cv2.imread(r"C:\FlowSync AI\test_media\test6.jpg")
    img_C = cv2.imread(r"C:\FlowSync AI\test_media\test5.jpg")

    if img_A is None or img_B is None or img_C is None:
        print("❌ Image missing")
        break

    # === COUNT ===
    countA = count_vehicles(img_A)
    countB = count_vehicles(img_B)
    countC = count_vehicles(img_C)

    # === DECISION ===
    lane_counts = {
        "A": countA,
        "B": countB,
        "C": countC
    }

    green_lane = max(lane_counts, key=lane_counts.get)

    # === UPDATE API ===
    latest_data["lane_A"] = countA
    latest_data["lane_B"] = countB
    latest_data["lane_C"] = countC
    latest_data["green"] = green_lane

    print(f"A:{countA} B:{countB} C:{countC} → GREEN: {green_lane}")

    time.sleep(2)  # simulate real-time updates