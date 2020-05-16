import tensorflow as tf
import keras
#from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

img_width = 200
img_height = 200
num_train_samples = 11000
num_validation_samples = 1100
epochs = 200
batch_size = 16

"""
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
"""

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data_images/train/',
    target_size=(200,200),
    color_mode='grayscale',
    #classes=['1F','2F','3F','4F','5F','CFist','FPalm','SPalm','Shaka','Rock1','Rock2'],
    class_mode='categorical',
    batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
    'data_images/test/',
    target_size=(200,200),
    color_mode='grayscale',
    #classes=['1F','2F','3F','4F','5F','CFist','FPalm','SPalm','Shaka','Rock1','Rock2'],
    class_mode='categorical',
    batch_size=batch_size)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(200, 200, 1))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dense(128, activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(11, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
        train_generator,
        steps_per_epoch=num_train_samples//batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=num_validation_samples//batch_size)

model.save("model7.h5")