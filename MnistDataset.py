import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

#print(tf.__version__)
#print(tf.test.gpu_device_name())

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[...,tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs=10,validation_data=(x_test, y_test))
model.save('my_model.h5')
#new_model = tf.keras.models.load_model('my_model.h5')


#loss,acc = new_model.evaluate(x_test,  y_test, verbose=2)
#print("Accuracy: {:5.2f}%".format(100*acc))
