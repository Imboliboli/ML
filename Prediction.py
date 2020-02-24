import numpy as np
import tensorflow as tf
from CatsAndDogs import train_generator
model = tf.keras.models.load_model('my_model2.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

img = tf.keras.preprocessing.image.load_img('C:/Users/fuyon/OneDrive/Desktop/Python/ML/test1.jpg', target_size=(150, 150))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images)
if classes[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(train_generator.class_indices)