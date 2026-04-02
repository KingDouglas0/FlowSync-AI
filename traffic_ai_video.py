from ultralytics import YOLO
import cv2
from flask import Flask, jsonify
import threading
import time
import logging

model = YOLO('yolov8n.pt')

app = Flask(__name__)

# ── Silence Flask request logs so they don't break the countdown ──────────────
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

latest_data = {
    "lane_A": 0,
    "lane_B": 0,
    "lane_C": 0,
    "green": "none",
    "time": 0,
    "cycle_position": 1,
    "total_cycles": 0
}

@app.route('/traffic', methods=['GET'])
def get_traffic():
    return jsonify(latest_data)

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
    return count, results

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return ret, frame

def show_frames(frameA, frameB, frameC, resA, resB, resC,
                countA, countB, countC, green_lane, remaining):
    dispA = resA[0].plot()
    dispB = resB[0].plot()
    dispC = resC[0].plot()

    for disp, lane, count in [(dispA, "A", countA),
                               (dispB, "B", countB),
                               (dispC, "C", countC)]:
        color = (0, 255, 0) if lane == green_lane else (0, 0, 255)
        cv2.putText(disp, f"Lane {lane}: {count} vehicles",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if lane == green_lane:
            cv2.putText(disp, f"GREEN  {remaining}s",
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "RED",
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Lane A", dispA)
    cv2.imshow("Lane B", dispB)
    cv2.imshow("Lane C", dispC)

    if cv2.waitKey(1) & 0xFF == 27:
        return False
    return True

# ── Open video sources ────────────────────────────────────────────────────────
cap_A = cv2.VideoCapture(r"C:\FlowSync AI\test_media\sample_traffic.mp4")
cap_B = cv2.VideoCapture(r"C:\FlowSync AI\test_media\sample_traffic2.mp4")
cap_C = cv2.VideoCapture(r"C:\FlowSync AI\test_media\sample_traffic3.mp4")

if not (cap_A.isOpened() and cap_B.isOpened() and cap_C.isOpened()):
    print("❌ One or more video files could not be opened. Check paths.")
    exit()

cycle_count = 0

while True:
    print("\n" + "="*50)
    print("🔍 Scanning all lanes...")
    print("="*50)

    retA, frameA = read_frame(cap_A)
    retB, frameB = read_frame(cap_B)
    retC, frameC = read_frame(cap_C)

    if not (retA and retB and retC):
        print("❌ Could not read frames")
        break

    countA, resA = count_vehicles(frameA)
    countB, resB = count_vehicles(frameB)
    countC, resC = count_vehicles(frameC)

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

        latest_data["green"] = lane
        latest_data["time"] = green_time
        latest_data["cycle_position"] = position + 1

        print(f"\n{priority_labels[position]}")
        print(f"  Lane {lane} → {count} vehicles → GREEN for {green_time}s",
              flush=True)

        for remaining in range(green_time, 0, -1):
            latest_data["time"] = remaining
            print(f"  ⏱  Lane {lane} GREEN — {remaining}s remaining   ",
                  end="\r", flush=True)

            _, frameA = read_frame(cap_A)
            _, frameB = read_frame(cap_B)
            _, frameC = read_frame(cap_C)

            _, resA = count_vehicles(frameA)
            _, resB = count_vehicles(frameB)
            _, resC = count_vehicles(frameC)

            keep_running = show_frames(
                frameA, frameB, frameC,
                resA, resB, resC,
                countA, countB, countC,
                lane, remaining
            )
            if not keep_running:
                print("\n\n👋 ESC pressed — exiting.")
                cap_A.release()
                cap_B.release()
                cap_C.release()
                cv2.destroyAllWindows()
                exit()

            time.sleep(1)

        print()
        latest_data["green"] = "none"
        print(f"  ✅ Lane {lane} done.", flush=True)

    print(f"\n🔄 Full cycle complete. Re-scanning now... (Total cycles: {cycle_count})")

cap_A.release()
cap_B.release()
cap_C.release()
cv2.destroyAllWindows()