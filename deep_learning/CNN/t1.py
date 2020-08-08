import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras_preprocessing import image

print(tf.__version__)

train_image = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_image.flow_from_directory("dataset/training_set",target_size=(64,64),batch_size=32,
                                               class_mode="binary")

print(training_set)
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory("dataset/test_set",target_size=(64,64),batch_size=32,
                                                         class_mode="binary")

cnn = Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=[64,64,1]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2))


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))

cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
cnn.fit(x=training_set,validation_data=test_set,epochs=15)


test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg",target_size=(64,64))

test_image= image.img_to_array(test_image)
test_image= np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0]==1:
    prediction="dog"
else:
    prediction="cat"
print(prediction)

