import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

path = "ML/fruits-360_dataset/fruits-360"

trainDir = os.path.join(path,"Training")
testDir = os.path.join(path,"Test")


train_datagen = ImageDataGenerator(rescale = 1.0/255)
testDir_datagen = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory(directory=trainDir, batch_size=128,
                                                    target_size=(100, 100),
                                                    color_mode="rgb",
                                                    class_mode="categorical",
                                                    shuffle=True)

test_generator = train_datagen.flow_from_directory(directory=trainDir, batch_size=128,
                                                    target_size=(100, 100),
                                                    color_mode="rgb",
                                                    class_mode="categorical",
                                                    shuffle=True)


model = keras.Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(100, 100, 3))
])


print(train_generator.class_indices)
