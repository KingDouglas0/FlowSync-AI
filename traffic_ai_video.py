from ultralytics import YOLO
import cv2
from flask import Flask, jsonify
import threading

# Load model
model = YOLO('yolov8n.pt')

# === STEP 1: 3 VIDEO SOURCES ===
cap_A = cv2.VideoCapture(r"C:\FlowSync AI\test_media\laneA.mp4")
cap_B = cv2.VideoCapture(r"C:\FlowSync AI\test_media\laneB.mp4")
cap_C = cv2.VideoCapture(r"C:\FlowSync AI\test_media\laneC.mp4")

# Flask setup
app = Flask(__name__)

# Store latest result
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

# === STEP 2: COUNT FUNCTION ===
def count_vehicles(frame):
    results = model(frame)
    boxes = results[0].boxes

    count = 0
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                count += 1

    return count, results


while True:
    # === STEP 3: READ ALL LANES ===
    retA, frameA = cap_A.read()
    retB, frameB = cap_B.read()
    retC, frameC = cap_C.read()

    if not (retA and retB and retC):
        print("End of video")
        break

    # === STEP 4: COUNT ===
    countA, resA = count_vehicles(frameA)
    countB, resB = count_vehicles(frameB)
    countC, resC = count_vehicles(frameC)

    # === STEP 5: DECISION ===
    lane_counts = {
        "A": countA,
        "B": countB,
        "C": countC
    }

    green_lane = max(lane_counts, key=lane_counts.get)

    # === STEP 6: UPDATE API ===
    latest_data["lane_A"] = countA
    latest_data["lane_B"] = countB
    latest_data["lane_C"] = countC
    latest_data["green"] = green_lane

    print(f"A:{countA} B:{countB} C:{countC} → GREEN: {green_lane}")

    # === STEP 7: DISPLAY ===
    frameA = resA[0].plot()
    frameB = resB[0].plot()
    frameC = resC[0].plot()

    cv2.putText(frameA, f"A: {countA}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frameB, f"B: {countB}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frameC, f"C: {countC}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Lane A", frameA)
    cv2.imshow("Lane B", frameB)
    cv2.imshow("Lane C", frameC)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap_A.release()
cap_B.release()
cap_C.release()
cv2.destroyAllWindows()