import cv2
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
from time import sleep

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

model = load_model('model8.h5')

x = []

"""
for i in range(11):
  img = cv2.imread('data_images/predict/'+str(i+1)+'.jpeg')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  x.append(img)

x = np.array(x, dtype="uint8")
x = x.reshape((11, 200, 200, 1))
"""

img = cv2.imread('data_images/predict/1.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x.append(img)
x = np.array(x, dtype="uint8")
x = x.reshape((1, 200, 200, 1))

y_pred = model.predict_classes(x)
#y_pred = np.argmax(y_pred, axis=1)
for i in range(len(y_pred)):
  y_pred[i] += 1
print(y_pred[0])