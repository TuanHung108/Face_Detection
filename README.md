# Face Detection Project - Phát Hiện Các Bộ Phận Khuôn Mặt

Dự án này sử dụng **YOLOv8** (You Only Look Once) để phát hiện và nhận diện các bộ phận khuôn mặt từ hình ảnh và video trong thời gian thực.

## 🎯 Mục Đích Dự Án

Phát hiện 3 bộ phận chính của khuôn mặt:
- **Eyes** (Mắt)
- **Nose** (Mũi)
- **Mouth** (Miệng)
Làm quen với việc sử dụng và fine-tune một model có sẵn
Theo dõi và đánh giá các thông số model như **Precision, Recall, F1 Score...**


## 📁 Cấu Trúc Dự Án

```
Face_Detection/
├── model.py                 # Training và prediction model
├── final.py                 # Real-time detection từ webcam
├── crop_frame_from_vid.py   # Tách frame từ video
├── auto_label.py            # Auto-labeling tạo file txt
├── output_json.py           # Export kết quả ra JSON
├── data.yaml                # Cấu hình dataset
├── trained_model.pt         # Model đã train
├── yolov8n.pt               # Model YOLOv8 nano (pre-trained)
├── face_dataset_1/          # Dataset gốc
│   ├── images/
│   │   ├── train/           # Training data
│   │   ├── val/             # Validation data
│   │   └── test/            # Test data
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── output/                  # Kết quả output
    ├── labels.json          # Labels dạng JSON
    └── txt_labels/          # Labels dạng TXT (YOLO format)
```

## ⚙️ Các File Chính

### 1. **model.py** - Training Model
Dùng để train model YOLOv8 trên dataset:
```python
python model.py
```

**Các tham số train:**
- Epochs: 50
- Image Size: 416x416
- Batch Size: 16
- Workers: 4

**Output:** `trained_model.pt` (model sau khi train)

### 2. **final.py** - Real-time Detection từ Webcam
Phát hiện khuôn mặt từ webcam theo thời gian thực:
```python
python final.py
```

**Tính năng:**
- Webcam real-time capture
- Vẽ bounding box cho mỗi bộ phận phát hiện
- Hiển thị độ tự tin (confidence score)
- Phân biệt màu sắc: 
  - Xanh lá (Eyes)
  - Xanh dương (Nose)
  - Đỏ (Mouth)

**Điều khiển:** Nhấn `Q` để thoát

### 3. **crop_frame_from_vid.py** - Tách Frame từ Video
Tách frame từ video để tạo dataset:
```python
extract_frames_from_video(input_folder, output_folder, frame_rate=2)
```

**Thông số:**
- `input_folder`: Thư mục chứa video
- `output_folder`: Nơi lưu frame
- `frame_rate`: Số frame/giây cần lưu (mặc định 1)

**Hỗ trợ định dạng:** MP4, AVI, MKV, MOV

### 4. **auto_label.py** - Tự động Gán Nhãn (TXT Format)
Tự động gán nhãn cho ảnh test sử dụng model đã train:
```python
python auto_label.py
```

**Output:** Tạo file `.txt` theo YOLO format (class_id x_center y_center width height)

### 5. **output_json.py** - Export Kết Quả ra JSON
Xuất kết quả phát hiện sang định dạng JSON:
```python
python output_json.py
```

**Output:** `output/labels.json` với cấu trúc:
```json
[
  {
    "image": "frame_0000.jpg",
    "objects": [
      {
        "label": "eyes",
        "bbox": [0.5, 0.3, 0.2, 0.15],
        "confidence": 0.95
      }
    ]
  }
]
```

## 📊 Dataset Configuration (data.yaml)

```yaml
train: face_dataset_1/images/train
val: face_dataset_1/images/val
test: face_dataset_1/images/test

nc: 3                           # Số class
names: ['Eyes', 'Nose', 'Mouth'] # Tên các class
```

## 🚀 Cách Sử Dụng

### 1. Chuẩn Bị Dataset
```bash
# Tách frame từ video
python crop_frame_from_vid.py
```

### 2. Train Model
```bash
python model.py
```

### 3. Real-time Detection
```bash
python final.py
```

### 4. Tự động Gán Nhãn (cho ảnh test)
```bash
python auto_label.py
```

### 5. Export Kết Quả
```bash
python output_json.py
```

## 📋 Requirements

- Python 3.8+
- OpenCV (cv2)
- Ultralytics YOLO
- PyYAML

**Install:**
```bash
pip install opencv-python ultralytics pyyaml
```

## 📈 Model Information

- **Base Model:** YOLOv8 Nano
- **Input Size:** 416x416
- **Classes:** 3 (Eyes, Nose, Mouth)
- **Training Epoch:** 50
- **Batch Size:** 16

## 📁 Output Formats

### YOLO TXT Format
```
<class_id> <x_center> <y_center> <width> <height>
```

**Ví dụ:**
```
0 0.5123 0.3456 0.1234 0.2345
1 0.6789 0.5432 0.1567 0.1823
```

### JSON Format
```json
{
  "image": "frame_0000.jpg",
  "objects": [
    {
      "label": "eyes",
      "bbox": [x_center, y_center, width, height],
      "confidence": 0.95
    }
  ]
}
```

## 🎨 Visualization

Real-time detection color coding:
- **Green (Xanh lá):** Eyes
- **Blue (Xanh dương):** Nose  
- **Red (Đỏ):** Mouth

## 📊 Training Results

Các kết quả training được lưu trong folder `runs/detect/`:
- `train/` - Training kỳ đầu
- `train2/` - Training kỳ thứ 2
- `train3/` - Training kỳ thứ 3
- `train4/` - Training kỳ thứ 4

Mỗi folder chứa:
- `weights/best.pt` - Model tốt nhất
- `weights/last.pt` - Model cuối cùng
- Biểu đồ huấn luyện (confusion_matrix, results.png)

## 🔍 Troubleshooting

### Webcam không mở được
```python
# Kiểm tra webcam device
cap = cv2.VideoCapture(0)  # Thay 0 bằng số khác nếu có nhiều camera
```

### Model không tìm thấy
Đảm bảo đường dẫn model đúng trong `final.py`:
```python
model = YOLO("runs/detect/train4/weights/best.pt")
```

### Memory không đủ
Giảm image size hoặc batch size trong `model.py`

## 📝 Ghi Chú

- Dự án sử dụng YOLOv8 Nano (model nhỏ, nhanh)
- Confidence threshold mặc định: 0.5
- Frame rate mặc định khi tách video: 2 frame/giây

