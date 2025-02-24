from keras.layers import Dense
from keras.layers import Input, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import SGD, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import GRU
from keras import utils
import numpy as np
import time
import argparse
from keras.layers import TimeDistributed
from keras.layers import InputSpec
from keras.layers import Layer
from matplotlib import pyplot as plt
import tensorflow as tf


class TemporalMaxPooling(Layer):
    def __init__(self, **kwargs):
        super(TemporalMaxPooling, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, x, mask=None):
        if mask is None:
            mask = K.sum(K.ones_like(x), axis=-1, keepdims=True)
        else:
            mask = K.expand_dims(mask)

        if K.backend() == "tensorflow":
            masked_data = x + (1.0 - K.cast(mask, K.floatx())) * -1e9
            return K.max(masked_data, axis=1)
        else:
            return K.max(x, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None



def rnn_models(model_name, train_data):
    main_input = Input(
        shape=(train_data.shape[1],
               train_data.shape[2]),
        name="main_input"
    )   

    if model_name == "lstm":
        headModel = LSTM(32)(main_input)

    elif model_name == "bidirectional":
        headModel = Bidirectional(LSTM(256, return_sequences=True))(main_input)
        headModel = LSTM(32)(headModel)

    elif model_name == "temporal_max":
        headModel = Bidirectional(LSTM(256, return_sequences=True))(main_input)
        headModel = TemporalMaxPooling()(headModel)

    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform"
    )(headModel)
    model = Model(inputs=main_input, outputs=predictions)

    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model


from keras.layers import Reshape

def lstm_model(train_data):
    # Model definition
    main_input = Input(
        shape=(train_data.shape[1], train_data.shape[2]),
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
    epochs = 15

    # Model compilation
    
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
    epochs=15
    batch_size=32
    weights_save_name="lstm_wts"


   

    # Training dataset loading
    train_data = np.load("lstm_40fpv_data.npy")
    train_label = np.load("lstm_40fpv_labels.npy")
    train_label = utils.to_categorical(train_label)
    print("Dataset Loaded...")

    # Train validation split
    trainX, valX, trainY, valY = train_test_split(
        train_data, train_label, shuffle=True, test_size=0.1
    )

    model = lstm_model(train_data)

    trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])


    # Number of trainable and non-trainable parameters
    print("Total params: {:,}".format(trainable_count + non_trainable_count))
    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))

    # Keras backend
    model_checkpoint = ModelCheckpoint(
        "trained_wts/" + weights_save_name + ".hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0)

    print("Training is going to start in 3... 2... 1... ")

    # Model training
    H = model.fit(
        trainX,
        trainY,
        validation_data=(valX, valY),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[model_checkpoint, stopping],
    )

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = stopping.stopped_epoch + 1
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots/training_plot.png")

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


if __name__ == '__main__':
    main()
