import cv2
import os
import glob

# Update the paths accordingly
training_videos_folder = ['train/0/', 'train/1/']

for folder in training_videos_folder:
    videos_path = glob.glob(os.path.join(folder, "*.mp4"))
    folder_name = os.path.basename(folder)

    for counter, video_path in enumerate(videos_path):
        cap = cv2.VideoCapture(video_path)
        vid = os.path.basename(video_path).split(".")[0]

        output_folder = f"/train_frames/{folder_name}/video_{counter}"
        os.makedirs(output_folder, exist_ok=True)

        frameId = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            filename = os.path.join(output_folder, f"image_{str(int(frameId) + 1)}.jpg")
            cv2.imwrite(filename, frame)
            frameId += 1

        cap.release()

        if counter % 100 == 0:
            print("Number of videos done:", counter)
