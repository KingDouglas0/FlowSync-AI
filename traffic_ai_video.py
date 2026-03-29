from ultralytics import YOLO
import cv2
from flask import Flask, jsonify
import threading

# Load model
model = YOLO('yolov8n.pt')

# Video source
cap = cv2.VideoCapture(r"C:\FlowSync AI\test_media\sample_traffic.mp4")

# Flask setup (Step 2)
app = Flask(__name__)
latest_density = "Low"

@app.route('/traffic', methods=['GET'])
def get_density():
    return jsonify({'density': latest_density})

def run_server():
    app.run(host='0.0.0.0', port=5000)

# Start Flask in background
threading.Thread(target=run_server).start()

# Vehicle classes (Step 1)
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    results = model(frame)

    boxes = results[0].boxes
    vehicle_count = 0

    # Count vehicles (Step 1)
    for box in boxes:
        cls = int(box.cls)
        if cls in vehicle_classes:
            vehicle_count += 1

    # Density logic (Step 1)
    if vehicle_count <= 5:
        density = "Low"
    elif vehicle_count <= 15:
        density = "Medium"
    else:
        density = "High"

    # Update API value (Step 2)
    latest_density = density

    print(f"Vehicles: {vehicle_count} | Density: {density}")

    annotated_frame = results[0].plot()

    # Display info
    cv2.putText(annotated_frame, f"Count: {vehicle_count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(annotated_frame, f"Density: {density}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Traffic Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()