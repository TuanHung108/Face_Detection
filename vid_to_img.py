import cv2

video_path = "C:\Code\Face_Detection\WIN_20241211_15_21_38_Pro.mp4"
output_folder = "C:/Code\Face_Detection/face_dataset_1/"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Tổng số khung hình: {frame_count}\nTốc độ khung hình: {fps}\nKích thước: {width}x{height}")

frame_rate = 10
frame_index = 0
saved_img = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % frame_rate == 0:
        file_name = f"{output_folder}frame{saved_img}.jpg"
        cv2.imwrite(file_name, frame)
        saved_img += 1

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

print(f"Lưu {saved_img} ảnh thành công")