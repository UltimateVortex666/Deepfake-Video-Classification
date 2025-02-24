from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.resnet50 import ResNet50
from keras.layers import Dropout
from keras.optimizers import Nadam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge
from keras.applications import EfficientNetB0, EfficientNetB5
from keras import backend as K
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import cv2
from keras.layers import LSTM, Bidirectional
import time
from keras.layers import InputSpec
from keras.layers import Layer
import argparse
import tensorflow as tf


def ignore_warnings(*args, **kwargs):
    pass


def cnn_model(model_name, img_size, weights):
    """
    Model definition using Xception net architecture
    """
    input_size = (int(img_size), int(img_size), 3)  # Convert img_size to int
    if model_name == "xception":
        baseModel = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=input_size  # Use input_size here
        )
    elif model_name == "iv3":
        baseModel = InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=input_size  # Use input_size here
        )
    elif model_name == "irv2":
        baseModel = InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_size  # Use input_size here
        )
    elif model_name == "resnet":
        baseModel = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_size  # Use input_size here
        )
    elif model_name == "nasnet":
        baseModel = NASNetLarge(
            weights="imagenet",
            include_top=False,
            input_shape=input_size  # Use input_size here
        )
    elif model_name == "ef0":
        baseModel = EfficientNetB0(
            weights="imagenet",
            include_top=False
        )
    elif model_name == "ef5":
        baseModel = EfficientNetB5(
            weights="imagenet",
            include_top=False
        )

    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(
        512,
        activation="relu",
        kernel_initializer="he_uniform",
        name="fc1")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    predictions = Dense(
        25,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

    model.load_weights(weights)
    print("Weights loaded...")
    model_lstm = Model(
        inputs=baseModel.input,
        outputs=model.get_layer("fc1").output
    )

    for layer in baseModel.layers:
        layer.trainable = True

    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model_lstm


def lstm_model(shape):
    # Model definition
    main_input = Input(
        shape=(shape[0],
               shape[1]),
        name="main_input"
    )
    headModel = Bidirectional(LSTM(256, return_sequences=True))(main_input)
    headModel = LSTM(32, return_sequences=True)(headModel)  # Add return_sequences=True
    headModel = TemporalMaxPooling()(headModel)
    headModel = Reshape((-1, 32))(headModel)  # Reshape to remove time dimension
    headModel = TimeDistributed(Dense(512))(headModel)
    headModel = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(headModel)  # Add return_sequences=True
    headModel = LSTM(256)(headModel)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform"
    )(headModel)
    model = Model(inputs=main_input, outputs=predictions)

    
    optimizer = Nadam(
        learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model


class TemporalMaxPooling(Layer):
    
    def __init__(self, **kwargs):
        super(TemporalMaxPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if mask is None:
            masked_data = K.max(x, axis=1)
        else:
            # Apply mask
            mask = K.expand_dims(mask, axis=-1)
            masked_data = K.max(x - (1 - mask) * 1e9, axis=1)  # Set masked values to a large negative value

        return masked_data

    def compute_mask(self, input, mask):
        # do not pass the mask to the next layers
        return None



def main():

    model_name="xception"
    image_size="160"
    load_weights_name="trained_wts\Saved_weights.hdf5"
    seq_lengths=25


    # MTCNN face extraction from frames
    imageio.core.util._precision_warn = ignore_warnings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    # Create face detector
    mtcnn = MTCNN(
        margin=40,
        select_largest=False,
        post_process=False,
        device=device
    )

    test_data = pd.read_csv('test_vids_label.csv')

    videos = test_data["vids_list"]
    true_labels = test_data["label"]

    # Loading model for feature extraction
    model = cnn_model(
        model_name=model_name,
        img_size=image_size,
        weights=load_weights_name
    )
    shape = (seq_lengths, 512)
    model_lstm = lstm_model(shape)

    model_lstm.load_weights("trained_wts\lstm_wts.hdf5")

    print("Weights loaded...")

    y_pred = []
    counter = 0

    for video in videos:
        cap = cv2.VideoCapture(video)
        batches = []

        while cap.isOpened() and len(batches) < seq_lengths:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            if h >= 720 and w >= 1080:
                frame = cv2.resize(
                    frame,
                    (640, 480),
                    interpolation=cv2.INTER_AREA
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            try:
                face = face.permute(1, 2, 0).int().numpy()
                batches.append(face)
            except AttributeError:
                print("Image Skipping")

        batches = np.asarray(batches).astype('float32')
        batches /= 255

        feature_vec = model.predict(batches)

        feature_vec = np.expand_dims(feature_vec, axis=0)

        preds = model_lstm.predict(feature_vec)
        y_pred += [preds[0].argmax(0)]

        cap.release()

        if counter % 10 == 0:
            print(counter, "Done....")
        counter += 1

    print("Accuracy Score:", accuracy_score(true_labels, y_pred))
    print("Precision Score", precision_score(true_labels, y_pred))
    print("Recall Score:", recall_score(true_labels, y_pred))
    print("F1 Score:", f1_score(true_labels, y_pred))

    np.save("lstm_preds.npy", y_pred)


if __name__ == '__main__':
    main()
