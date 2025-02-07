import os
import cv2
import json
from ultralytics import YOLO

names = ['Eyes', 'Nose', 'Mouth']
label_map = {name: idx for idx, name in enumerate(names)}

def auto_labeling(input_folder, output_json):
    model = YOLO("C:/Code/Face_Detection/runs/detect/train3/weights/best.pt")
    
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    results = []

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape

        predictions = model.predict(img, conf=0.5)[0]
        objects = []

        for pred in predictions.boxes:
            x_min, y_min, x_max, y_max = pred.xyxy[0].tolist()
            conf = pred.conf[0].item()
            label = names[int(pred.cls[0].item())]

            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            objects.append({
                "label": label.lower(),
                "bbox": [round(x_center, 4), round(y_center, 4), round(width, 4), round(height, 4)],
                "confidence": round(conf, 4)
            })

        results.append({
            "image": img_file,
            "objects": objects
        })

    with open(output_json, 'w') as json_file:
        json.dump(results, json_file, indent=4)



input_folder = "C:/Code/Face_Detection/face_dataset_1/images/test"
output_json_file = "C:/Code/Face_Detection/output/labels.json"
os.makedirs(os.path.dirname(output_json_file), exist_ok = True)

auto_labeling(input_folder, output_json_file)





# Bước 1: Dataset đã gán nhãn (ground truth).
# Bước 2: Auto-labeling trên dataset đó.
# Bước 3: Tinh IoU & Accuracy để so sánh với bbox của ground truth.
# Bước 4: Tính Precision, Recall, F1-score.
# Bước 5: Vẽ biểu đồ.