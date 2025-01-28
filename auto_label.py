import os
import json
import cv2
from ultralytics import YOLO

names = ['Eyes', 'Nose', 'Mouth']
label_map = {name: idx for idx, name in enumerate(names)}

def auto_labeling(input_folder, output_txt_folder):
    model = YOLO("C:/Code/Face_Detection/runs/detect/train3/weights/best.pt")
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    results = []

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape

        predictions = model.predict(img, conf=0.5)[0]
        detections = []

        txt_path = os.path.join(output_txt_folder, img_file.replace('.jpg', '.txt'))
        with open(txt_path, 'w') as f:
            for pred in predictions.boxes:
                x_min, y_min, x_max, y_max = pred.xyxy[0].tolist()
                conf = pred.conf[0].item()
                label = names[int(pred.cls[0].item())]

                class_id = label_map[label]

                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h


                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    classes_file = os.path.join(output_txt_folder, "classes.txt")
    with open(classes_file, 'w') as f:
        for name in names:
            f.write(name + '\n')


input_folder = "C:/Code/Face_Detection/face_dataset/images/val"
output_txt_folder = "C:/Code/Face_Detection/output/txt_labels"
os.makedirs(output_txt_folder, exist_ok = True)

auto_labeling(input_folder, output_txt_folder)