import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input

# Define the CNN model function
def cnn_model(img_size):
    input_size = (img_size, img_size, 3)
    base_model = Xception(weights="imagenet", include_top=False, input_shape=input_size)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu", kernel_initializer="he_uniform")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    predictions = Dense(25, activation="softmax", kernel_initializer="he_uniform")(x)  # Assuming 25 output classes

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    return model

# Define function to load and preprocess frames
def load_and_preprocess_frame(frame_path, target_size):
    img = Image.open(frame_path)
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# Define function to compute GradCAM
# Define function to compute GradCAM
def compute_gradcam(model, img_array, class_index):
    conv_output = model.get_layer('block14_sepconv2_act').output
    model_end = Model(inputs=model.input, outputs=[conv_output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = model_end(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.reduce_mean(tf.multiply(conv_output, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap  # Return the heatmap directly without calling .numpy()

# Rest of the code remains unchanged...


# Define the main function
def main():
    # Create and load the CNN model
    model = cnn_model(img_size=160)
    model.load_weights("trained_wts/Saved_weights.hdf5")
    print("Model weights loaded successfully.")

    # Load test videos and process frames
    test_data = pd.read_csv("test_vids_label.csv")
    videos = test_data["vids_list"][:4]  # Select the first 4 videos for processing

    # Process each video
    for counter, video_path in enumerate(videos):
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to match model input size (160x160)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, (160, 160))
            frame_array = img_to_array(frame_resized)
            frame_array = preprocess_input(frame_array[np.newaxis])

            # Compute GradCAM for the predicted class
            preds = model.predict(frame_array)
            class_idx = np.argmax(preds[0])
            heatmap = compute_gradcam(model, frame_array, class_idx)

            # Resize heatmap to match original frame size
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Overlay heatmap on original frame
            overlaid_frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

            # Display overlaid frame
            cv2.imshow("GradCAM", overlaid_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
