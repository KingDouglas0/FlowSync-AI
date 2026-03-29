from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(r"C:\FlowSync AI\test_media\sample_traffic.mp4")  # use a traffic video

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    results = model(frame)

    annotated_frame = results[0].plot()  # draws boxes

    cv2.imshow("Traffic Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to stop
        break

cap.release()
cv2.destroyAllWindows()