import keras
from keras import backend as k
from keras.preprocessing import ImageDataGenerator

img_width = 200
img_height = 200
num_train_samples = 1100
nb_validation_samples = 110
epochs = 50
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_it = datagen.flow_from_directory(
    'data_images/train/',
    target_size=(200,200),
    color_mode='grayscale',
    classes=['1F','2F','3F','4F','5F','CFist','FPalm','SPalm','Shaka','Rock1','Rock2'],
    class_mode='categorical',
    batch_size=batch_size)

val_it = datagen.flow_from_directory(
    'data_images/test/',
    target_size=(200,200),
    color_mode='grayscale',
    classes=['1F','2F','3F','4F','5F','CFist','FPalm','SPalm','Shaka','Rock1','Rock2'],
    class_mode='categorical',
    batch_size=batch_size)