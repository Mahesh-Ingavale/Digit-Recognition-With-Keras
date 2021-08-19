# Digit-Recognition-With-Keras 
Digit Recognition [ Artificial Neural Network ]
  
 #Part 1
 #Import Requirements
 import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

#Part 2
(X_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data()

#Part 3
len(X_train)

#Part 4
len(X_test)

#Part 5
X_train[0].shape

#Part 6
X_train[0]

#Part 7
plt.matshow(X_train[0])

#Part 8
y_train[2]

#Part 9
y_train[:5]

#Part 10
X_train.shape

#Part 11
X_train = X_train / 255
X_test = X_test / 255
X_train[0]

#Part 12
X_train_flattened = X_train.reshape(len(X_train),28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)

#Part 13
X_test_flattened.shape

#Part 14
X_train_flattened[0]

#Part 15
model =  keras.Sequential([
      keras.layers.Dense(10,input_shape=(784,), activation="sigmoid")
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train_flattened, y_train, epochs=5)

#Part 16
model.evaluate(X_train_flattened, y_train)

#Part 17
plt.matshow(X_test[1])

#Part 18
y_predicted = model.predict(X_test_flattened)
y_predicted[1]

#Part 19
np.argmax(y_predicted[1])

#Part 20
y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]

#Part 21
y_test[:5]

#Part 22
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
cm

#Part 23
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

#Part 24
model =  keras.Sequential([
      keras.layers.Dense(100, input_shape=(784,), activation="relu"),
      keras.layers.Dense(10, activation="sigmoid")

])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train_flattened, y_train, epochs=5)

#Part 25
model.evaluate(X_test_flattened, y_test)

#Part 26
#Graphical Representation
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')

#Part 27
model =  keras.Sequential([
      keras.layers.Flatten(input_shape=(28,28)),
      keras.layers.Dense(100, activation="relu"),
      keras.layers.Dense(10, activation="sigmoid")

])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=5)
plt.ylabel('Truth')
