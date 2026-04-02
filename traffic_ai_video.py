from ultralytics import YOLO
import cv2
from flask import Flask, jsonify
import threading
import time
import logging

model = YOLO('yolov8n.pt')

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

latest_data = {
    "lane_A": 0,
    "lane_B": 0,
    "lane_C": 0,
    "green": "none",
    "yellow": "none",   # ← new: tracks which lane is yellow
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
                countA, countB, countC, green_lane, yellow_lane, remaining):
    dispA = resA[0].plot()
    dispB = resB[0].plot()
    dispC = resC[0].plot()

    for disp, lane, count in [(dispA, "A", countA),
                               (dispB, "B", countB),
                               (dispC, "C", countC)]:

        if lane == green_lane:
            color = (0, 255, 0)       # green text
            status = f"GREEN  {remaining}s"
        elif lane == yellow_lane:
            color = (0, 255, 255)     # yellow text
            status = "YELLOW"
        else:
            color = (0, 0, 255)       # red text
            status = "RED"

        cv2.putText(disp, f"Lane {lane}: {count} vehicles",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(disp, status,
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Lane A", dispA)
    cv2.imshow("Lane B", dispB)
    cv2.imshow("Lane C", dispC)

    if cv2.waitKey(1) & 0xFF == 27:
        return False
    return True

def yellow_transition(lane, cap_A, cap_B, cap_C,
                      countA, countB, countC,
                      resA, resB, resC,
                      yellow_duration=3):
    """
    Holds yellow for yellow_duration seconds after a green phase ends.
    Keeps video windows live during the transition.
    Returns False if ESC pressed, True otherwise.
    """
    print(f"\n  🟡 Lane {lane} — YELLOW ({yellow_duration}s transition)",
          flush=True)

    # Update API so ESP32 knows yellow is active
    latest_data["green"] = "none"
    latest_data["yellow"] = lane
    latest_data["time"] = 0

    for t in range(yellow_duration, 0, -1):
        print(f"  🟡 Lane {lane} YELLOW — {t}s   ", end="\r", flush=True)

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
            green_lane="none",   # no green during yellow
            yellow_lane=lane,
            remaining=0
        )
        if not keep_running:
            return False

        time.sleep(1)

    print()
    latest_data["yellow"] = "none"
    return True

# ── Open video sources ─────────────────────────────────────────────────────────
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
        latest_data["yellow"] = "none"
        latest_data["time"] = green_time
        latest_data["cycle_position"] = position + 1

        print(f"\n{priority_labels[position]}")
        print(f"  Lane {lane} → {count} vehicles → GREEN for {green_time}s",
              flush=True)

        # ── Green phase countdown ──────────────────────────────────────────
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
                green_lane=lane,
                yellow_lane="none",
                remaining=remaining
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
        print(f"  ✅ Lane {lane} green phase done.", flush=True)

        # ── Yellow transition before next lane ────────────────────────────
        keep_running = yellow_transition(
            lane, cap_A, cap_B, cap_C,
            countA, countB, countC,
            resA, resB, resC,
            yellow_duration=3      # ← change this to adjust yellow time
        )
        if not keep_running:
            print("\n\n👋 ESC pressed — exiting.")
            cap_A.release()
            cap_B.release()
            cap_C.release()
            cv2.destroyAllWindows()
            exit()

    print(f"\n🔄 Full cycle complete. Re-scanning now... (Total cycles: {cycle_count})")

cap_A.release()
cap_B.release()
cap_C.release()
cv2.destroyAllWindows()