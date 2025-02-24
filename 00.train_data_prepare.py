import cv2
import numpy as np
import tensorflow as tf
import glob
from os.path import join
from os import listdir
from random import shuffle
from sklearn.preprocessing import LabelEncoder

# Manually set the values
img_size = 160
frames_per_video = 25

# Ensure train_path contains at least one element
train_path = ["train_face/"]

list_1 = [join(train_path[0], x) for x in listdir(train_path[0])]

c = 0

label_encoder = LabelEncoder()

for i in range(len(list_1)):
    vid_list = list_1[i * frames_per_video: (i + 1) * frames_per_video]
    shuffle(vid_list)

    train_data1 = []
    train_label1 = []

    count = 0

    for x in vid_list:
        img = glob.glob(join(x, "*.jpg"))
        img.sort(key=lambda f: int("".join(filter(str.isdigit, f))) if any(char.isdigit() for char in f) else 0)
        images = img[:frames_per_video]
        labels = [k.split("/")[1] for k in img][:frames_per_video]

        if count % 10 == 0:
            print("Number of files done:", count)
        count += 1

        # Convert string labels to integers
        encoded_labels = label_encoder.fit_transform(labels)

        for j, k in zip(images, encoded_labels):
            img = cv2.imread(j)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            train_data1.append(img)
            train_label1.append(k)

            if count % 10 == 0:
                print("Number of files done:", count)
            count += 1

    train_data1 = np.array(train_data1)
    train_label1=np.array(train_label1)

    
    
    # Check if train_data is non-empty before saving
    if train_data1.size > 0:
        train_label = tf.keras.utils.to_categorical(train_label1)
        print(train_data1.shape, train_label1.shape)
        np.save(f"./train_data1_{i}.npy", train_data1)
        np.save(f"./train_label1_{i}.npy", train_label1)

        print("Files saved for iteration", i)
