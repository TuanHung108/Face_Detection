import cv2
import os

def extract_frames_from_video(input_folder, output_folder_root, frame_rate=1):
    """
    Tách frame từ tất cả video trong thư mục và lưu vào các thư mục riêng.

    Args:
        input_folder (str): Thư mục chứa video.
        output_folder_root (str): Thư mục gốc để lưu các frame.
        frame_rate (int): Số frame mỗi giây cần lưu (mặc định 1 frame/giây).
    """
    # Đảm bảo thư mục output gốcốc tồn tại
    if not os.path.exists(output_folder_root):
        os.makedirs(output_folder_root)

    # Duyệt qua các fole trong thư mục input
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)

        # Kiểm tra nếu file là video (dựa trên phần mở rộng)
        if not video_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            continue
        
        # Tạo thư mục output riêng cho từng video
        video_name = os.path.splitext(video_file)[0]
        output_folder_video = os.path.join(output_folder_root, video_name)
        if not os.path.exist(output_folder_video):
            os.makedirs(output_folder_video)

        # Mở videovideo
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Cannot open video: {video_file}")
            continue

        # Lấy FPS (số frame/second) từ videovideo
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / frame_rate) # Số frame cần bỏ qua trước khi lưu một frame (khoảng cách giữa các frame)

        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()  # ret trả về False nếu hết video
            if not ret:
                break
            
            # Lưu frame nếu nó nằm trong khoảng thời gian cần lưu
            if frame_count % frame_interval == 0:
                frame_name = os.path.join(output_folder_video, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_name, frame)
                print(f"Save: {frame_name}")
                saved_count += 1

            frame_count += 1

        # Giải phóng tài nguyênnguyên
        cap.release()
        print(f"{saved_count} frame from video '{video_file}' and save to folder '{output_folder_video}'.")



input_folder = ""  # Thay bằng đường dẫn đến thư mục chứa video
output_folder_root = ""  # Đường dẫn đến nơi chứa thư mực mưu frame
extract_frames_from_video(input_folder, output_folder_root, frame_rate=2)