import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import cv2
import time
import torch
import numpy as np
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.optimizers import Nadam
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications import EfficientNetB5, EfficientNetB0
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def ignore_warnings(*args, **kwargs):
    pass

def cnn_model(model_name, img_size):
    input_shape=(img_size, img_size, 3)
    if model_name == "xception":
        baseModel = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "iv3":
        baseModel = InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "irv2":
        baseModel = InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "resnet":
        baseModel = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "nasnet":
        baseModel = NASNetLarge(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "ef0":
        baseModel = EfficientNetB0(
            weights="imagenet",  # Add this line to specify weights
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "ef5":
        baseModel = EfficientNetB5(
            weights="imagenet",  # Add this line to specify weights
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    # headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
    #     headModel
    # )
    # headModel = Dropout(0.5)(headModel)
    headModel = Dropout(0.5)(headModel)
    predictions = Dense(
        25,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

    for layer in baseModel.layers:
        layer.trainable = True

    optimizer = Nadam(
        learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model



def main():
    start = time.time()

    epochs = 25
    model_name = "xception"
    weights_save_name = "Saved_weights"
    batch_size = 32
    image_size = 160
    num_classes = 25

    # Read video labels from csv file
    test_data = pd.read_csv("test_vids_label.csv")
    

    videos = test_data["vids_list"]
    true_labels = test_data["label"]

    # Suppress unnecessary warnings
    imageio.core.util._precision_warn = ignore_warnings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create face detector
    mtcnn = MTCNN(
        margin=40,
        select_largest=True,
        post_process=False,
        device=device
    )

    # Loading model weights
    model = cnn_model(model_name, img_size=image_size)
    model.load_weights("trained_wts/" + weights_save_name + ".hdf5")

    print("Weights loaded...")

    y_predictions = []
    y_probabilities = []
    videos_done = 0
    for video in videos:
        cap = cv2.VideoCapture(video)
        batches = []

        # Number of frames taken into consideration for each video
        while (cap.isOpened() and len(batches) < 25):
            ret, frame = cap.read()
            if not ret:
                break

            # Rest of your processing code...
            frame = cv2.resize(frame, (image_size, image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            # Check if face is detected
            face = mtcnn(frame)
            if face is not None:
                face = face.permute(1, 2, 0).int().numpy()
                batches.append(face)
            else:
                print("Face not detected in the current frame.")

        if len(batches) > 0:
            batches = np.asarray(batches).astype("float32")
            batches /= 255

            predictions = model.predict(batches)
            predictions_mean = np.mean(predictions, axis=0)
            y_probabilities += [predictions_mean]
            y_predictions += [predictions_mean.argmax(0)]
        else:
            print("No frames found for video:", video)

        cap.release()

        if videos_done % 10 == 0:
            print("Number of videos done:", videos_done)
        videos_done += 1

    print("Accuracy Score:", accuracy_score(true_labels, y_predictions))
    print("Precision Score:", precision_score(true_labels, y_predictions))
    print("Recall Score:", recall_score(true_labels, y_predictions))
    print("F1 Score:", f1_score(true_labels, y_predictions))

    # Saving predictions and probabilities for further calculation
    # of AUC scores.
    np.save("Y_predictions.npy", y_predictions)
    np.save("Y_probabilities.npy", y_probabilities)

    end = time.time()
    dur = end - start

    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur > 60 and dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")


if __name__ == "__main__":
    main()
