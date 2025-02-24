from keras.layers import Activation, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input, ZeroPadding3D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.regularizers import l2
import numpy as np
import cv2
import os
import random
import time
import matplotlib.pyplot as plt
from os.path import join
from os import listdir
from random import shuffle

def conv3d_model(batch_size):
    input_shape = (batch_size, 112, 112, 3)
    weight_decay = 0.005
    nb_classes = 2

    inputs = Input(input_shape)
    x = Conv3D(
        64,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(inputs)
    x = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding="same")(x)

    x = Conv3D(
        128,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = Conv3D(
        128,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = Conv3D(
        256,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = Conv3D(
        256,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = Flatten()(x)
    x = Dense(2048, activation="relu", kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu", kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, kernel_regularizer=l2(weight_decay))(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x)
    return model
def c3d_model(batch_size):
    """ Return the Keras model of the network
    """
    main_input = Input(shape=(batch_size, 112, 112, 3), name="main_input")
    # 1st layer group
    x = Conv3D(
        64,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv1",
        strides=(1, 1, 1),
    )(main_input)
    x = MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1, 2, 2), padding="valid", name="pool1"
    )(x)
    # 2nd layer group
    x = Conv3D(
        128,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv2",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool2"
    )(x)
    # 3rd layer group
    x = Conv3D(
        256,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv3a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        256,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv3b",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool3"
    )(x)
    # 4th layer group
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv4a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv4b",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool4"
    )(x)
    # 5th layer group
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv5a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv5b",
        strides=(1, 1, 1),
    )(x)
    x = ZeroPadding3D(padding=(0, 1, 1))(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool5"
    )(x)
    x = Flatten()(x)
    # FC layers group
    x = Dense(2048, activation="relu", name="fc6")(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu", name="fc7")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax", name="fc8")(x)

    model = Model(inputs=main_input, outputs=predictions)
    return model



def plot_history(history, result_dir):
    plt.plot(history.history["accuracy"], marker=".")
    plt.plot(history.history["loss"], marker=".")
    plt.title("model")
    plt.grid()
    plt.legend(["accuracy", "loss"], loc="lower right")
    plt.savefig(os.path.join(result_dir, "model.png"))
    plt.close()


def save_history(history, result_dir):
    loss = history.history["loss"]
    acc = history.history["accuracy"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_accuracy"]
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, "result.txt"), "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write(
                "{}\t{}\t{}\t{}\t{}\n".format(
                    i, loss[i], acc[i], val_loss[i], val_acc[i]
                )
            )


def process_batch(video_paths, batch_size, train=True):
    num = len(video_paths)
    batch = np.zeros((num, batch_size, 112, 112, 3), dtype="float32")
    labels = np.zeros(num, dtype="int")
    for i in range(num):
        path = video_paths[i]
        
        # Split the path and check if it results in at least two parts
        parts = path.split("/")
        if len(parts) < 2:
            # Handle the case where the split operation doesn't result in enough parts
            continue
        
        label = parts[1]
        label = int(label)
    
        imgs = os.listdir(path)
        imgs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)

            for j in range(batch_size):
                img = imgs[j]
                image = cv2.imread(path + "/" + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                if is_flip == 1:
                    image = cv2.flip(image, 1)
                batch[i][j][:][:][:] = image[
                    crop_x: crop_x + 112, crop_y: crop_y + 112, :
                ]
            labels[i] = label
        else:
            for j in range(batch_size):
                img = imgs[j]
                image = cv2.imread(path + "/" + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                batch[i][j][:][:][:] = image[8:120, 30:142, :]
            labels[i] = label
    return batch, labels




def preprocess(inputs):
    inputs /= 255.0
    return inputs


def generator_batch(vid_list, batch_size, num_classes, train=True):
    num_samples = len(vid_list)
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0:
        num_batches += 1
    while True:
        shuffle(vid_list)

        for i in range(0, num_samples, batch_size):
            batch_paths = vid_list[i:i + batch_size]
            x_batch, y_batch = process_batch(batch_paths, batch_size, train=train)
            x = preprocess(x_batch)
            y = to_categorical(y_batch, num_classes)
            yield x, y


def main():
    start = time.time()
    epochs = 15
    batch_size = 32
    weights_save_name = "Cnn"
    model_name = "c3d"

    # Video cropped faces train path
    train_path = "train_face"  # Adjust this to your specific folder
    vid_list = [join(train_path, x) for x in listdir(train_path)]

    train_vid_list = vid_list[:int(0.8 * len(vid_list))]
    val_vid_list = vid_list[int(0.8 * len(vid_list)):]

    num_classes = 2

    if model_name == "c3d":
        model = conv3d_model(batch_size=batch_size)
    else:
        model = conv3d_model(batch_size=batch_size)

    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    history = model.fit(
    generator_batch(train_vid_list, batch_size, num_classes, train=True),
    steps_per_epoch=len(train_vid_list) // batch_size,
    epochs=epochs,
    validation_data=generator_batch(val_vid_list, batch_size, num_classes, train=False),
    validation_steps=len(val_vid_list) // batch_size,
    verbose=1,
)


    result_dir = "results/"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    plot_history(history, result_dir)
    save_history(history, result_dir)
    model.save_weights(join(result_dir, weights_save_name + ".hdf5"))

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
