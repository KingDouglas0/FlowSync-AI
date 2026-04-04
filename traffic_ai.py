from ultralytics import YOLO
import cv2
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import threading
import time
import logging

model = YOLO('yolov8n.pt')

app = Flask(__name__)
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

latest_data = {
    "lane_A": 0,
    "lane_B": 0,
    "lane_C": 0,
    "green": "none",
    "yellow": "none",
    "time": 0,
    "cycle_position": 1,
    "total_cycles": 0
}

extension_used = {
    "A": True,
    "B": False,
    "C": False
}

EXTENSION_SECONDS = 10

# ── Route 1: dashboard page ───────────────────────────────────────────────────
@app.route('/')
def dashboard():
    return send_file('dashboard.html')

# ── Route 2: traffic data JSON ────────────────────────────────────────────────
@app.route('/traffic', methods=['GET'])
def get_traffic():
    return jsonify(latest_data)

# ── Route 3: IR sensor extension ─────────────────────────────────────────────
@app.route('/extend', methods=['POST'])
def extend_green():
    data = request.get_json()
    if not data or "lane" not in data:
        return jsonify({"status": "error", "reason": "missing lane"}), 400

    lane = data["lane"].upper()

    if lane not in ["B", "C"]:
        return jsonify({"status": "error",
                        "reason": "no IR sensor on that lane"}), 400

    if latest_data["green"] != lane:
        return jsonify({"status": "ignored",
                        "reason": f"lane {lane} is not currently green"}), 200

    if extension_used[lane]:
        return jsonify({"status": "ignored",
                        "reason": f"extension already used for lane {lane} this cycle"}), 200

    extension_used[lane] = True
    latest_data["time"] += EXTENSION_SECONDS
    print(f"\n  ⚡ IR sensor on Lane {lane} triggered — +{EXTENSION_SECONDS}s added",
          flush=True)

    return jsonify({
        "status": "extended",
        "lane": lane,
        "added_seconds": EXTENSION_SECONDS,
        "new_time": latest_data["time"]
    }), 200

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

threading.Thread(target=run_server, daemon=True).start()

vehicle_classes = [2, 3, 5, 7]

def get_green_time(count):
    if count <= 5:
        return 25
    elif count <= 15:
        return 40
    else:
        return 70

def count_vehicles(frame):
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    count = 0
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                count += 1
    return count

def yellow_phase(lane, next_lane, yellow_duration=3):
    """
    Two-stage yellow transition between lanes.
    Stage 1 — outgoing lane goes yellow.
    Stage 2 — incoming lane goes yellow (if there is one).
    """
    # Stage 1 — outgoing yellow
    print(f"\n  🟡 Lane {lane} — YELLOW (clearing)", flush=True)
    latest_data["green"]  = "none"
    latest_data["yellow"] = lane

    for t in range(yellow_duration, 0, -1):
        latest_data["time"] = t
        print(f"  🟡 Outgoing Lane {lane} YELLOW — {t}s   ",
              end="\r", flush=True)
        time.sleep(1)

    print()
    latest_data["yellow"] = "none"

    # Stage 2 — incoming yellow
    if next_lane is not None:
        print(f"\n  🟡 Lane {next_lane} — YELLOW (preparing)", flush=True)
        latest_data["yellow"] = next_lane

        for t in range(yellow_duration, 0, -1):
            latest_data["time"] = t
            print(f"  🟡 Incoming Lane {next_lane} YELLOW — {t}s   ",
                  end="\r", flush=True)
            time.sleep(1)

        print()
        latest_data["yellow"] = "none"

# Image pool — rotates each cycle to simulate changing traffic
image_pool = [
    r"C:\FlowSync AI\test_media\test5.jpg",
    r"C:\FlowSync AI\test_media\test6.jpg",
    r"C:\FlowSync AI\test_media\test7.jpg",
]

cycle_count = 0

while True:
    print("\n" + "="*50)
    print("🔍 Scanning all lanes...")
    print("="*50)

    # Reset extension flags every cycle
    extension_used["A"] = True
    extension_used["B"] = False
    extension_used["C"] = False

    # Rotate image assignments each cycle so counts vary
    offset = cycle_count % len(image_pool)
    path_A = image_pool[(0 + offset) % len(image_pool)]
    path_B = image_pool[(1 + offset) % len(image_pool)]
    path_C = image_pool[(2 + offset) % len(image_pool)]

    img_A = cv2.imread(path_A)
    img_B = cv2.imread(path_B)
    img_C = cv2.imread(path_C)

    if img_A is None or img_B is None or img_C is None:
        print("❌ Image missing — check file paths")
        break

    print("  Counting Lane A...", flush=True)
    countA = count_vehicles(img_A)

    print("  Counting Lane B...", flush=True)
    countB = count_vehicles(img_B)

    print("  Counting Lane C...", flush=True)
    countC = count_vehicles(img_C)

    print(f"📊 Scan results → A:{countA}  B:{countB}  C:{countC}")

    latest_data["lane_A"] = countA
    latest_data["lane_B"] = countB
    latest_data["lane_C"] = countC
    cycle_count += 1
    latest_data["total_cycles"] = cycle_count

    lane_counts = {"A": countA, "B": countB, "C": countC}
    sorted_lanes = sorted(lane_counts.items(), key=lambda x: x[1], reverse=True)
    priority_labels = ["🟢 HIGH (1st)", "🟡 MEDIUM (2nd)", "🔴 LOW (3rd)"]

    for position, (lane, count) in enumerate(sorted_lanes):
        green_time = get_green_time(count)
        next_lane  = sorted_lanes[position + 1][0] \
                     if position + 1 < len(sorted_lanes) else None

        latest_data["green"]          = lane
        latest_data["yellow"]         = "none"
        latest_data["time"]           = green_time
        latest_data["cycle_position"] = position + 1

        print(f"\n{priority_labels[position]}")
        print(f"  Lane {lane} → {count} vehicles → GREEN for {green_time}s",
              flush=True)

        # Green phase countdown
        # Uses latest_data["time"] directly so /extend can modify it
        while latest_data["time"] > 0:
            remaining = latest_data["time"]
            print(f"  ⏱  Lane {lane} GREEN — {remaining}s remaining   ",
                  end="\r", flush=True)
            time.sleep(1)
            latest_data["time"] -= 1

        print()
        print(f"  ✅ Lane {lane} green phase done.", flush=True)

        # Two-stage yellow transition
        yellow_phase(lane, next_lane, yellow_duration=3)

    print(f"\n🔄 Full cycle complete. Re-scanning now... (Total cycles: {cycle_count})")