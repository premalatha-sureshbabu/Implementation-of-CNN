# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset
Develop a model that can classify images of handwritten digits (0-9) from the MNIST dataset with high accuracy. The model should use a convolutional 
neural network architecture and be optimized using early stopping to avoid overfitting.

## Neural Network Model


Include the neural network model diagram.(http://alexlenail.me/NN-SVG/index.html)

## DESIGN STEPS

### Step 2:
Load the MNIST dataset, which contains images of handwritten digits and corresponding labels.

### Step 3:
Reshape and normalize the images by scaling the pixel values between 0 and 1.

### Step 4:
Define a CNN model using layers such as convolution, pooling, and fully connected layers. Add a softmax output layer for classification.

### Step 5:
Compile the model using an appropriate optimizer (e.g., Adam), loss function (e.g., categorical crossentropy), and evaluation metric (accuracy).

### Step 6:
Train the model on the training dataset for a fixed number of epochs (e.g., 10 epochs) while monitoring the validation accuracy using early 
stopping to halt training once a desired accuracy is achieved.

### Step 7:
Stop training if the model reaches 98% accuracy on the training set to prevent overfitting and save computation time.


## PROGRAM

### Name: Prema Latha S
### Register Number: 212222230112

### Importing Libraries
```
import os
import base64
import numpy as np
import tensorflow as tf
```
### Loading and Inspecting the MNIST dataset
```

# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "/content/mnist.npz.zip")

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
```

### Resizing and Reshaping Function
```
FUNCTION: reshape_and_normalize

def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """

    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = np.expand_dims(images, axis=-1)

    # Normalize pixel values
    images = images/255.0

    ### END CODE HERE ###

    return images
```
```
# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print('Name: Meetha Prabhu            RegisterNumber: 212222240065         \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")
```
### CallBack Function
```

def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """

    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = np.expand_dims(images, axis=-1)

    # Normalize pixel values
    images = images/255.0

    ### END CODE HERE ###

    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print('Name: S.Prema Latha           RegisterNumber: 212222230112         \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")
```
## GRADED FUNCTION: convolutional_model

### Network Model
```
def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    ## START CODE HERE ###

    # Define the model
    model = tf.keras.models.Sequential([

    # Add convolutions and max pooling
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Add the same layers as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

    ### END CODE HERE ###

    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
    return model
```
### Model Summary:
```
model.summary()
```
```
### Model compiling and Training

# Define your compiled (but untrained) model
model = convolutional_model()

# Train your model (this can take up to 5 minutes)
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])
```
## OUTPUT

### Reshape and Normalize output
![Screenshot 2024-09-16 133928](https://github.com/user-attachments/assets/09f83134-e0f5-4647-b174-49635c7ffc59)


### Model Summary
![Screenshot 2024-09-16 115224](https://github.com/user-attachments/assets/8c22fb58-8283-4609-9b7a-a4fdc4b82158)


### Training the model output
![Screenshot 2024-09-16 115218](https://github.com/user-attachments/assets/e3cddec9-34f2-4dbb-81f8-bda21c960db2)




## RESULT
Thus the program to create a Convolution Neural Network to classify images is successfully implemented.
