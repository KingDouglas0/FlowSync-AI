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
    "yellow": "none",
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

def get_latest_frames(cap_A, cap_B, cap_C):
    """Read one fresh frame from each lane and run detection."""
    _, fA = read_frame(cap_A)
    _, fB = read_frame(cap_B)
    _, fC = read_frame(cap_C)
    _, rA = count_vehicles(fA)
    _, rB = count_vehicles(fB)
    _, rC = count_vehicles(fC)
    return fA, fB, fC, rA, rB, rC

def show_frames(frameA, frameB, frameC, resA, resB, resC,
                countA, countB, countC, green_lane, yellow_lane, remaining):
    dispA = resA[0].plot()
    dispB = resB[0].plot()
    dispC = resC[0].plot()

    for disp, lane, count in [(dispA, "A", countA),
                               (dispB, "B", countB),
                               (dispC, "C", countC)]:
        if lane == green_lane:
            color  = (0, 255, 0)
            status = f"GREEN  {remaining}s"
        elif lane == yellow_lane:
            color  = (0, 255, 255)
            status = "YELLOW"
        else:
            color  = (0, 0, 255)
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

def yellow_phase(outgoing_lane, incoming_lane,
                 cap_A, cap_B, cap_C,
                 countA, countB, countC,
                 resA, resB, resC,
                 yellow_duration=3):
    """
    Two-stage yellow transition between lanes:

    Stage 1 — OUTGOING yellow:
        The lane that just finished green goes yellow.
        All other lanes stay red.
        ESP32 shows yellow on outgoing lane.

    Stage 2 — INCOMING yellow:
        The lane about to go green goes yellow.
        All other lanes stay red.
        ESP32 prepares drivers that green is coming.
    """

    # ── Stage 1: outgoing lane yellow ─────────────────────────────────────
    print(f"\n  🟡 Lane {outgoing_lane} — YELLOW (clearing)", flush=True)
    latest_data["green"]  = "none"
    latest_data["yellow"] = outgoing_lane

    for t in range(yellow_duration, 0, -1):
        latest_data["time"] = t
        print(f"  🟡 Outgoing Lane {outgoing_lane} YELLOW — {t}s   ",
              end="\r", flush=True)

        fA, fB, fC, rA, rB, rC = get_latest_frames(cap_A, cap_B, cap_C)

        keep = show_frames(fA, fB, fC, rA, rB, rC,
                           countA, countB, countC,
                           green_lane="none",
                           yellow_lane=outgoing_lane,
                           remaining=0)
        if not keep:
            return False
        time.sleep(1)

    print()
    latest_data["yellow"] = "none"

    # ── Stage 2: incoming lane yellow (only if there is a next lane) ───────
    if incoming_lane is not None:
        print(f"\n  🟡 Lane {incoming_lane} — YELLOW (preparing)", flush=True)
        latest_data["yellow"] = incoming_lane

        for t in range(yellow_duration, 0, -1):
            latest_data["time"] = t
            print(f"  🟡 Incoming Lane {incoming_lane} YELLOW — {t}s   ",
                  end="\r", flush=True)

            fA, fB, fC, rA, rB, rC = get_latest_frames(cap_A, cap_B, cap_C)

            keep = show_frames(fA, fB, fC, rA, rB, rC,
                               countA, countB, countC,
                               green_lane="none",
                               yellow_lane=incoming_lane,
                               remaining=0)
            if not keep:
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

        # Determine which lane comes next (None if this is the last lane)
        next_lane = sorted_lanes[position + 1][0] if position + 1 < len(sorted_lanes) else None

        latest_data["green"]         = lane
        latest_data["yellow"]        = "none"
        latest_data["time"]          = green_time
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

        # ── Two-stage yellow transition ────────────────────────────────────
        # outgoing = current lane, incoming = next lane (None on last lane)
        keep_running = yellow_phase(
            outgoing_lane=lane,
            incoming_lane=next_lane,
            cap_A=cap_A, cap_B=cap_B, cap_C=cap_C,
            countA=countA, countB=countB, countC=countC,
            resA=resA, resB=resB, resC=resC,
            yellow_duration=3
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