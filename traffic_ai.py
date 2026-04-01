from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# === STEP 1: LOAD 3 LANE IMAGES ===
img_A = cv2.imread(r"C:\FlowSync AI\test_media\test7.jpg")
img_B = cv2.imread(r"C:\FlowSync AI\test_media\test6.jpg")
img_C = cv2.imread(r"C:\FlowSync AI\test_media\test5.jpg")

# Check images
if img_A is None or img_B is None or img_C is None:
    print("❌ One or more images NOT found. Check paths.")
    exit()
else:
    print("✅ All images loaded successfully!")

# Vehicle classes (COCO)
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


# === STEP 3: COUNT EACH LANE ===
countA, resA = count_vehicles(img_A)
countB, resB = count_vehicles(img_B)
countC, resC = count_vehicles(img_C)

# === STEP 4: DECISION ===
lane_counts = {
    "A": countA,
    "B": countB,
    "C": countC
}

green_lane = max(lane_counts, key=lane_counts.get)

# === STEP 5: PRINT RESULT ===
print(f"A: {countA}, B: {countB}, C: {countC}")
print(f"🚦 GREEN LIGHT → Lane {green_lane}")

# === STEP 6: DISPLAY RESULTS ===
frameA = resA[0].plot()
frameB = resB[0].plot()
frameC = resC[0].plot()

cv2.putText(frameA, f"A: {countA}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

cv2.putText(frameB, f"B: {countB}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

cv2.putText(frameC, f"C: {countC}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# Show windows
cv2.imshow("Lane A", frameA)
cv2.imshow("Lane B", frameB)
cv2.imshow("Lane C", frameC)

cv2.waitKey(0)
cv2.destroyAllWindows()