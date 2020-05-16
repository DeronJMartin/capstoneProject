import cv2
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
import pynput
from pynput.keyboard import Key, Controller

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    e = ''

gesture_dict = {
    1: 'One Finger',
    2: 'Two Fingers',
    3: 'Three Fingers',
    4: 'Four Fingers',
    5: 'Five Fingers',
    6: 'Closed Fist',
    7: 'Front Palm',
    8: 'Side Palm',
    9: 'Shaka',
    10: 'Rock1',
    11: 'Rock2',
    12: 'None'}

model = load_model('model12.h5')

keyboard = Controller()

cap = cv2.VideoCapture(0)

prev_gesture = -1

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    kernel = np.ones((3,3), dtype=np.uint8)

    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 5)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, kernel, iterations = 4)
    mask = cv2.GaussianBlur(mask, (5,5), 100)

    cv2.imshow('Frame',frame)
    cv2.imshow('Mask',mask)

    mask = mask.reshape((1,200,200,1))

    prediction = model.predict_classes(x)
    gesture = prediction + 1
    print(gesture_dict[gesture])

    if ((gesture != prev_gesture) and (prev_gesture != -1)):
        if (gesture == 1):
            keyboard.press(Key.media_volume_down)
        elif (gesture == 2):
            keyboard.press(Key.media_previous)
        elif (gesture == 3):
            keyboard.press(Key.media_play_pause)
        elif (gesture == 4):
            keyboard.press(Key.media_next)
        elif (gesture ==5):
            keyboard.press(Key.media_volume_up)
        elif (gesture == 12):
            print("No Gesture Detected")
        else:
            print("Incorrect Gesture")

    prev_gesture = gesture
