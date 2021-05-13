import tensorflow as tf
#tensorflow.random.set_seed(112)

import numpy as np


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd


datagen = ImageDataGenerator()


train_it = datagen.flow_from_directory('drive/MyDrive/data/train', class_mode="categorical", shuffle=False, batch_size=64, target_size=(256, 256)) #directory is to my google drive
test_it = datagen.flow_from_directory('drive/MyDrive/data/test', class_mode="categorical", shuffle=False, batch_size=64, target_size=(256, 256))
val_it = datagen.flow_from_directory('drive/MyDrive/data/valid', class_mode="categorical", shuffle=False, batch_size=64, target_size=(256, 256))


test_it.image_shape



#IMAGE_HEIGHT = 256
#IMAGE_WIDTH = 256
#COLOR_CHANNELS = 3
#IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
#TOTAL_INPUTS = IMAGE_PIXELS * COLOR_CHANNELS


#Build Sequential Model

model = Sequential()
#model.add(Dropout(0.2, input_shape=(TOTAL_INPUTS,)))
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(256, 256, 3), activation='relu',))


#1st Convolutional Block
model.add(Conv2D(10,(5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

# 2nd conv block
model.add(Conv2D(20, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

# 3rd conv block
#model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
#model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
#model.add(BatchNormalization())

# ANN block
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))

# output layer
model.add(Dense(units=2, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])



model.summary()


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# fit on data for 10 epochs
model.fit_generator(train_it, epochs=10, validation_data=val_it,callbacks=[early_stop])


model.history.history.keys()


metrics = pd.DataFrame(model.history.history)


metrics


metrics[['loss', 'val_loss']].plot()
plt.title('Training Loss Vs Validation Loss', fontsize=16)
plt.show()


metrics[['accuracy', 'val_accuracy']].plot()
plt.title('Training Accuracy Vs Validation Accuracy', fontsize=16)
plt.show()


model.metrics_names


# Formatting the result: "test_loss" to 3 decimal places, while "test_Accuracy" to percentage 1 decimal place

test_loss, test_accuracy = model.evaluate(test_it)
print(f'Test loss is {test_loss:0.3} and test accuracy is {test_accuracy:0.1%}')
