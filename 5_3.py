import cv2
import os
import keras
from keras.models import load_model
import tensorflow as tf
import json
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

model = load_model('model14.h5')

while (True):
    sleep(0.2)

    mask = cv2.imread('live/live_mask.jpeg')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.reshape((1, 200, 200, 1))

    prediction = model.predict_classes(mask)
    gesture = prediction[0] + 1

    os.chdir('live')
    gesture_file = open('live_gesture.txt', 'w')
    gesture_file.write(str(gesture))
    print("Writing: " + str(gesture))
    gesture_file.close()
    os.chdir('..')