from facenet_pytorch import MTCNN
import torch
import cv2
from PIL import Image
from os import listdir, makedirs
import glob
from os.path import join, exists
import imageio.core.util

def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings

# Check if GPU (CUDA) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create face detector
mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device=device
)

# Directory containing images respective to each video
source_frames_folders = ["C:/train_frames/"]
# Destination location where faces cropped out from images will be saved
dest_faces_folder = "./train_face/0/"

for i in source_frames_folders:
    counter = 0
    for j in listdir(i):
        imgs = glob.glob(join(i, j, "*.jpg"))
        if counter % 1000 == 0:
            print("Number of videos done:", counter)
        if not exists(join(dest_faces_folder, j)):
            makedirs(join(dest_faces_folder, j))
        for k in imgs:
            frame = cv2.imread(k)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            
            # Check if face is detected
            face = mtcnn(frame)
            if face is not None:
                try:
                    print(f"Saving face from {k}")
                    face.save(join(dest_faces_folder, j, k.split("/")[-1]))
                except AttributeError:
                    print("Image skipping due to AttributeError")
            else:
                print(f"No face detected in {k}")
                
        counter += 1