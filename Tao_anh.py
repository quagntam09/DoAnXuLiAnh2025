import cv2
import os

def extract_frames_from_video(video_path, output_folder, frame_interval=30):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            
            cv2.imwrite(filename, frame)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    my_video = "video_mau.mp4" 
    
    output_dir = "sub_img" 
    
    extract_frames_from_video(my_video, output_dir, frame_interval=30)