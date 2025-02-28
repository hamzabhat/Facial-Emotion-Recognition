import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix

# Unzipping data
#!unzip archive.zip

"""## Retrieving image attributes"""

train_directory = " "
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
for emotion in classes:
    path = os.path.join(train_directory, emotion)
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break

IMG_SIZE = 224
new_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resizing
print(new_image.shape)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

"""## Creating an image array and resizing"""

# Data array
train_array = []

def create_training_Data():
    for category in classes:
        path = os.path.join(train_directory, category)
        class_num = classes.index(category)  # 0 1 2 ... 6, #LABEL
        for img in os.listdir(path):
            try:
                image_path = os.path.join(path, img)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read {image_path}")
                    continue
                new_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resizing every image
                train_array.append([new_image, class_num])
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

create_training_Data()  # Function Call
print(len(f"Number of images{train_array}"))

random.shuffle(train_array)  # Unbiased modeling so that model doesn't learn based on the sequence of the images

"""## Separating features and labels"""

X = []  # Feature
Y = []  # Label
for features, label in train_array:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # Converting images to 4D since MobileNet has depth-wise neural networks that take 4 dimensions
X.shape

"""## NORMALIZATION"""
# X = X / 255

"""# MODEL TRAINING"""

model = tf.keras.applications.MobileNetV2()  # Pre-trained model
model.summary()

"""## Transfer Learning"""

base_input = model.layers[0].input  # Input layer
base_output = model.layers[-2].output  # Selecting the global_average_pooling2d_2 as the output layer
print(base_output)

"""### Adding new dense layers to the network"""

final_output = Dense(128)(base_output)  # Adding a new layer (128 nodes) after base_output layer
final_output = layers.Activation('relu')(final_output)
final_output = Dense(64)(final_output)  # Adding a new layer (64 nodes)
final_output = layers.Activation('relu')(final_output)
final_output = Dense(7, activation='softmax')(final_output)  # Final output layer (7 nodes) uses softmax activation function
print(final_output)

"""### Creating Model"""

new_model = keras.Model(inputs=base_input, outputs=final_output)
new_model.summary()
new_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""### Training the new model"""

# Y = np.array(Y)
# new_model.fit(X, Y, epochs=25)

"""### Saving & Loading the new model"""

# new_model.save('FER_transfer_model.h5')
model = tf.keras.models.load_model('F_FER_transfer_model.h5')

"""# TESTING"""

IMG_SIZE = 224
test_directory = " "
test_array = []

def create_test_Data():
    for category in classes:
        path = os.path.join(test_directory, category)
        class_num = classes.index(category)  # 0 1 2 ... 6, #LABEL
        for img in os.listdir(path):
            try:
                image_path = os.path.join(path, img)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read {image_path}")
                    continue
                new_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resizing every image
                test_array.append([new_image, class_num])
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

create_test_Data()  # Function Call
print(len(f"Number of images{test_array}"))

X_test = []  # Feature
y_test = []  # Label
for features, label in test_array:
    X_test.append(features)
    y_test.append(label)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # Converting images to 4D
X_test.shape
y_test = np.array(y_test)

"""#### Model Characteristics"""

# RESULTS
y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_test)))
plt.xticks(tick_marks, np.unique(y_test))
plt.yticks(tick_marks, np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
pyplot.savefig("confusion_matrix_mobilenet.png")

test_accu = np.sum(y_test == y_pred) / len(y_test) * 100
print(f"test accuracy: {round(test_accu, 4)} %\n\n")

print(classification_report(y_test, y_pred))
