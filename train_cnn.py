import argparse
import numpy as np
import cv2 
import keras
from keras import layers
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--output", required=True,
	help="path to output model after training")
args = vars(ap.parse_args())

class digit_classifier:
  @staticmethod
  def build(width, height, depth, classes):
    input_shape=(width, height, depth)
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(384, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(classes, activation="softmax"),
    ])
    print(model.summary()) 
    return model


print("[INFO] accessing MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Scale data to the [0, 1] range
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Add a channel dimension to the digits (Make sure images have shape (28, 28, 1))
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# convert the labels from integers to vectors
num_classes=10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# compile the network

print("------------ Compiling model ------------")
model = digit_classifier.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the network
print("------------ Training model ------------")
batch_size = 128
epochs = 10

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,validation_split=0.1)

# evaluate the network
print("------------ Evaluating model ------------")
predictions = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1),	predictions.argmax(axis=1),	target_names=[str(x) for x in range(10)]))
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# serialize the model to disk
print("------------ Serializing model ------------")
model.save(args["model"], save_format="h5")

