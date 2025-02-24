import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from keras.utils import to_categorical

def cnn_model(model_name, img_size, num_classes):
    input_size = (img_size, img_size, 3)
    model_dict = {
        "xception": tf.keras.applications.Xception,
        "iv3": tf.keras.applications.InceptionV3,
        "irv2": tf.keras.applications.InceptionResNetV2,
        "resnet": tf.keras.applications.ResNet50,
        "nasnet": tf.keras.applications.NASNetLarge,
        "ef0": tf.keras.applications.EfficientNetB0,
        "ef5": tf.keras.applications.EfficientNetB5
    }

    baseModel = model_dict[model_name](
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
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

    # Directly pass arguments
    epochs = 25
    model_name = "xception"
    weights_save_name = "Saved_weights"
    batch_size = 32
    image_size = 224
    num_classes = 25

    os.makedirs("./trained_wts", exist_ok=True)
    os.makedirs("./training_logs", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    # Replace the next two lines with your actual data loading code
    train_data = np.load("train_data.npy")
    train_label = np.load("train_label.npy")
    # Assuming train_data and train_label are your actual data
    trainX, valX, trainY, valY = train_test_split(
        train_data, train_label, test_size=0.1, shuffle=False
    )
    print("Length of training data:", len(trainX))
    print(trainX.shape)
    print(valX.shape)
    print(trainY.shape)
    print(valY.shape)

    trainAug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valAug = ImageDataGenerator(rescale=1.0 / 255.0)

    model = cnn_model(model_name, img_size=image_size, num_classes=num_classes)

    trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])

    print(f"Total params: {trainable_count + non_trainable_count:,}")
    print(f"Trainable params: {trainable_count:,}")
    print(f"Non-trainable params: {non_trainable_count:,}")

    model_checkpoint = ModelCheckpoint(
        f"trained_wts/{weights_save_name}.hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    csv_logger = CSVLogger("training_logs/xception.log", separator=",", append=True)

    print("Training is going to start in 3... 2... 1... ")
    trainY_onehot = trainY.astype(int)
    valY_onehot = valY.astype(int)
    print(trainX.shape)
    print(valX.shape)
    print(trainY_onehot.shape)
    print(valY_onehot.shape)

    H = model.fit(
        trainAug.flow(trainX, trainY_onehot, batch_size=batch_size),
        validation_data=valAug.flow(valX,valY_onehot),
        validation_steps=len(valX),
        epochs=epochs,
        callbacks=[model_checkpoint, csv_logger],
    )

    # Plotting section
    plt.style.use("ggplot")
    plt.figure()
    epochs_range = np.arange(0, len(H.history["loss"]))
    plt.plot(epochs_range, H.history["loss"], label="train_loss")
    plt.plot(epochs_range, H.history["accuracy"], label="train_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots/training_plot.png")

    end = time.time()
    dur = end - start

    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")

if __name__ == "__main__":
    main()
