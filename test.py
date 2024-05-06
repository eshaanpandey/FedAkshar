# check tensorflow gpu availability
import tensorflow as tf
import pickle
import numpy as np
# defining Train and Test data path
test_path = r'C:\Users\dell5\OneDrive\Desktop\archive\DevanagariHandwrittenCharacterDataset\Test'

# generating test and train data with tensorflow

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_path, image_size=(32, 32), batch_size=40, label_mode='categorical', shuffle=True, interpolation="lanczos5")

# defining class names and printing them
class_names = test_data.class_names
""" print(class_names)
 """
# import the necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# create the model object
model = Sequential()

# add the first convolutional layer
model.add(Convolution2D(32, (3,3), strides=1, activation="relu", input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

# add the second convolutional layer
model.add(Convolution2D(32, (3,3), strides=1, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

# add the third convolutional layer
model.add(Convolution2D(64, (3,3), strides=1, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

# add the fourth convolutional layer
model.add(Convolution2D(64, (3,3), strides=1, activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

# flatten the output of the last convolutional layer
model.add(Flatten())

# add the first dense layer with weight decay regularization
model.add(Dense(128, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# add the second dense layer with weight decay regularization
model.add(Dense(64, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# add the output layer with softmax activation function
model.add(Dense(46, activation="softmax", kernel_initializer="uniform"))

# Load aggregated weights from file
with open("aggregated_weights.pickle", "rb") as f:
    aggregated_weights = pickle.load(f)

# Set weights to the model
model.set_weights(aggregated_weights)


# get the predicted classes
predicted_probs = model.predict(test_data)
predicted_classes = np.argmax(predicted_probs, axis=1)

# get the true classes
true_classes = np.concatenate([y for x, y in test_data], axis=0)
true_classes = np.argmax(true_classes, axis=1)

# get the class labels
class_labels = list(test_data.class_names)




# predictig on 32 test images
import cv2
import numpy as np
import matplotlib.pyplot as plt

def predict_and_plot_images(model, class_names, test_data):
    # get a batch of test images and labels
    test_images, test_labels = next(iter(test_data))

    # make predictions on the batch of test images
    predictions = model.predict(test_images)

    # get the predicted class for each image in the batch
    predicted_classes = np.argmax(predictions, axis=1)

    # plot the images and their predicted labels
    num_images = len(test_images)
    num_rows = (num_images // 5) + 1
    fig, axs = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
    axs = axs.flatten()

    for i in range(num_images):
        # convert the image from a tensor to a numpy array
        image = test_images[i].numpy()

        # convert the one-hot encoded label to an integer label
        true_label = np.argmax(test_labels[i])

        # get the predicted label and confidence
        predicted_label = predicted_classes[i]
        confidence = predictions[i][predicted_label]

        # plot the image and the true/predicted labels
        axs[i].imshow(image)
        axs[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[predicted_label]}\nConfidence: {confidence:.2f}',
                         fontsize=8, pad=2)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

predict_and_plot_images(model, class_names, test_data)

# Function to pridict on a single random image

import cv2
import numpy as np
import matplotlib.pyplot as plt

def predict_and_display(image_path, model, class_names):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to (32, 32)
    resized_image = cv2.resize(image, (32, 32))

    # Convert the image to a numpy array
    image_array = np.array(resized_image)

    # Expand the dimensions of the image array to match the input shape of the model
    image_array = np.expand_dims(image_array, axis=0)

    # Make the prediction
    prediction = model.predict(image_array)

    # Get the predicted class label with the highest probability
    predicted_class = class_names[np.argmax(prediction)]
    
    # Get the predicted probability
    confidence = np.max(prediction)

    # Display the image, predicted label, and confidence
    plt.imshow(image[:,:,::-1]) # Convert BGR to RGB
    plt.axis('off')
    plt.title(f'Predicted label: {predicted_class}\nConfidence: {confidence:.2f}')
    plt.show()


#image_path = '/kaggle/input/hindi-test-sample-letters/gha.png'
# you can cange the image_path to your custom hindi handwritten letter image
# or can test the provided samples at hindi-test-sample-letters
#predict_and_display(image_path, model, class_names)




