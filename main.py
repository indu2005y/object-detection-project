from ultralytics import YOLO
import cv2

# YOLO model load
model = YOLO("yolov8n.pt")

# Webcam open
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()