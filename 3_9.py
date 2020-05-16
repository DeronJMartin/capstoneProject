import cv2
import numpy as np
import os
import re
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

#Natural Sorting Function
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

#Training Data Directory Definition
directory = '' + os.getcwd() + '/data_images/train'
os.chdir(directory)

x = []
y = []

for i in range(11):
    for j in range(1000):
        img = cv2.imread(sorted_alphanumeric(os.listdir())[i]+'/'+str(j+1)+'.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x.append(img)
        y.append(i+1)

x = np.array(x, dtype="uint8")
x = x.reshape((11000, 200, 200, 1))
y = np.array(y)

labelencoder_y_1 = LabelEncoder()
y = labelencoder_y_1.fit_transform(y)

ts = 0.25

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=ts, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(11, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

os.chdir('..')
os.chdir('..')
model.save("model9.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy: {:2.2f}%'.format(test_acc*100))