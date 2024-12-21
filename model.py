from ultralytics import YOLO

# Training model
def train_model():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        workers=4
    )
    model.save("trained_model.pt")

# Predict with training model
def predict_with_model():
    model = YOLO("trained_model.pt")
    results = model.predict(
        source="face_dataset_1/images/val",  
        imgsz=416,  
        conf=0.5   
    )
    for result in results:
        result.show()

if __name__ == "__main__":
    train_model()
    predict_with_model()
