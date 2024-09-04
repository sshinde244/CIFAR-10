import keras #deep learning API for implementing neural networks
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np #used for working with arrays

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Initialize the model
model = Sequential()
# Add the first convolutional layer
#Conv2D: 3x3 kernel, applied on image of 32x32 pixels
#relu: function that returns 0 for neagtive input, returns back positive value
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# Add the first max pooling layer
#MaxPooling2D: downsamples input along spatial dimensions
model.add(MaxPooling2D((2, 2)))
# Add the second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
# Add the second max pooling layer
model.add(MaxPooling2D((2, 2)))
# Flatten the output
#Flattent(): converts resultant 2D arrays from pooling into linear vector
model.add(Flatten())
# Add the first fully connected layer
#Dense: layer to classify images based on convolving output
model.add(Dense(128, activation='relu'))
# Add the output layer
model.add(Dense(10, activation='softmax'))
# Compile the model
#adam: gradiant descent method
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size= 64, epochs=15, validation_data=(x_test, y_test), shuffle=True)

# Evaluate the model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
