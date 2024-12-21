import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open Webcam!")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cannot recognize frame.")
        break

    # Predict with YOLO model
    results = model(frame)

    for box in results[0].boxes:
        x1, y1, x2, y2, = map(int, box.xyxy[0])
        conf = box.conf[0]
        label = int(box.cls[0])

        if (label == 0):
            frame = cv2. rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, f'{results[0].names[label]} {conf:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if (label == 1):
            frame = cv2. rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frame = cv2.putText(frame, f'{results[0].names[label]} {conf:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if (label == 2):
            frame = cv2. rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            frame = cv2.putText(frame, f'{results[0].names[label]} {conf:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()