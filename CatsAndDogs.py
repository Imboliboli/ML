import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

PATH = r"C:\Users\fuyon\PycharmProjects\ML\Images"
trainDir = os.path.join(PATH, "train")
valDir = os.path.join(PATH, "validation")
train_cat = os.path.join(trainDir, "cats")
train_dog = os.path.join(trainDir, "dogs")
val_cat = os.path.join(valDir, "cats")
val_dog = os.path.join(valDir, "dogs")

print(len(os.listdir(train_cat)))
print(len(os.listdir(train_dog)))
print(len(os.listdir(val_cat)))
print(len(os.listdir(val_dog)))

train_datagen = ImageDataGenerator(
                    rescale=1./255
                    # rotation_range=45,
                    # width_shift_range=.15,
                    # height_shift_range=.15,
                    # horizontal_flip=True,
                    # zoom_range=0.5
                    )
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(directory=trainDir, batch_size=128,
                                                    target_size=(150, 150),
                                                    color_mode="rgb",
                                                    class_mode="binary",
                                                    shuffle=True)

val_generator = val_datagen.flow_from_directory(directory=valDir, batch_size=128,
                                                target_size=(150, 150),
                                                color_mode="rgb",
                                                class_mode="binary")

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(),
    # Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    # Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=2000 / 128, epochs=1, validation_data=val_generator,
                              validation_steps=1000 / 128)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epoch = range(len(loss))



plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(epoch, loss, label='loss')
plt.plot(epoch, val_loss, label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epoch, acc, label='acc')
plt.plot(epoch, val_acc, label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

model.save('my_model2.h5')
new_model = tf.keras.models.load_model('my_model2.h5')
loss,acc = new_model.evaluate(val_generator, verbose=2)
print("Accuracy: {:5.2f}%".format(100*acc))
