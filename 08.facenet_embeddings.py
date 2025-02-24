import cv2
import numpy as np
import pandas as pd
import glob
import os
from os.path import isfile, join, split, splitext
from random import shuffle
from keras.preprocessing import image
from keras_facenet import FaceNet

# Function to extract frames from videos
def extract_frames(video_path, target_size=(160, 160), num_frames=25):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_frames:
        print(f"Video {video_path} has less frames than required. Skipping.")
        return None
    frame_indices = sorted([int(x * (frame_count - 1) / (num_frames - 1)) for x in range(num_frames)])
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            frame = cv2.resize(frame, target_size)  # Resize frame
            frames.append(frame)
    cap.release()
    return frames

# Read csv file containing image and video paths
data = pd.read_csv("train_files.csv")

file_paths = data["images_list"]
labels = data["label"]

train_data = []
train_label = []

# Check if a checkpoint file exists
checkpoint_file = "checkpoint.txt"
start_index = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        start_index = int(f.read().strip())

embedder = FaceNet()

for idx, (file_path, label) in enumerate(zip(file_paths[start_index:], labels[start_index:]), start=start_index):
    try:
        print(f"Processing file {idx + 1}/{len(file_paths)}: {file_path}")
        _, ext = splitext(file_path)
        if ext.lower() in ['.jpg', '.jpeg', '.png']:
            # Load the image using Keras' image.load_img() function
            img = image.load_img(file_path, target_size=(160, 160))

            # Convert the image to an array
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Get the embeddings using the embedder
            embeddings = embedder.embeddings(x)

            train_data.append(embeddings)
            train_label.append(label)  # Just append the label as it is, assuming it's already in the correct format

        else:
            # Extract frames from the video
            frames = extract_frames(file_path)
            if frames is not None:
                for frame in frames:
                    print(f"Processing frame of video: {file_path}")
                    # Convert frame to an array
                    x = np.expand_dims(frame, axis=0)
                    
                    # Get the embeddings using the embedder
                    embeddings = embedder.embeddings(x)
                    
                    train_data.append(embeddings)
                    train_label.append(label)  # Just append the label as it is, assuming it's already in the correct format
    except Exception as e:
        print(f"Error processing file: {file_path}. Skipping...")
        print(e)

    # Save checkpoint after processing every 100 files
    if (idx + 1) % 1000 == 0:
        with open(checkpoint_file, "w") as f:
            f.write(str(idx + 1))

train_data = np.array(train_data)
train_label = np.array(train_label)

np.save("train_data_facenet_embeddings.npy", train_data)
np.save("train_label_facenet_embeddings.npy", train_label)
print("Files saved.")
