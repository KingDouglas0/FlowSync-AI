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
    "green": "none",
    "time": 0,
    "cycle_position": 1,   # 1=highest, 2=medium, 3=lowest
    "total_cycles": 0
}

@app.route('/traffic', methods=['GET'])
def get_traffic():
    return jsonify(latest_data)

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# Start Flask server in background
threading.Thread(target=run_server, daemon=True).start()

# Vehicle classes (COCO)
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

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

def get_green_time(count):
    if count <= 5:
        return 25
    elif count <= 15:
        return 40
    else:
        return 70

cycle_count = 0

while True:
    print("\n" + "="*50)
    print("🔍 Scanning all lanes...")
    print("="*50)

    # === LOAD IMAGES ===
    img_A = cv2.imread(r"C:\FlowSync AI\test_media\test6.jpg")
    img_B = cv2.imread(r"C:\FlowSync AI\test_media\test7.jpg")
    img_C = cv2.imread(r"C:\FlowSync AI\test_media\test5.jpg")

    if img_A is None or img_B is None or img_C is None:
        print("❌ Image missing")
        break

    # === COUNT VEHICLES ===
    countA = count_vehicles(img_A)
    countB = count_vehicles(img_B)
    countC = count_vehicles(img_C)

    print(f"📊 Scan results → A:{countA}  B:{countB}  C:{countC}")

    # === UPDATE COUNTS IN API ===
    latest_data["lane_A"] = countA
    latest_data["lane_B"] = countB
    latest_data["lane_C"] = countC
    cycle_count += 1
    latest_data["total_cycles"] = cycle_count

    # === SORT LANES: highest → medium → lowest ===
    lane_counts = {"A": countA, "B": countB, "C": countC}
    sorted_lanes = sorted(lane_counts.items(), key=lambda x: x[1], reverse=True)

    priority_labels = ["🟢 HIGH (1st)", "🟡 MEDIUM (2nd)", "🔴 LOW (3rd)"]

    # === RUN THE FULL PRIORITY CYCLE ===
    for position, (lane, count) in enumerate(sorted_lanes):
        green_time = get_green_time(count)

        # Update shared API data
        latest_data["green"] = lane
        latest_data["time"] = green_time
        latest_data["cycle_position"] = position + 1

        print(f"\n{priority_labels[position]}")
        print(f"  Lane {lane} → {count} vehicles → GREEN for {green_time}s")

        # Countdown display while lane is green
        for remaining in range(green_time, 0, -1):
            latest_data["time"] = remaining
            print(f"  ⏱  Lane {lane} GREEN — {remaining}s remaining", end="\r")
            time.sleep(1)

        print(f"\n  ✅ Lane {lane} done.")

    print(f"\n🔄 Full cycle complete. Re-scanning now... (Total cycles: {cycle_count})")