import cv2
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
import pynput
from pynput.keyboard import Key, Controller
from time import sleep

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

mask_num = -1

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

    mask_num += 1

    cv2.imshow('Frame',frame)
    cv2.imshow('Mask',mask)

    mask = mask.reshape((1,200,200,1))

    if (mask_num % 60 == 59):
        prediction = model.predict_classes(mask)
        gesture = prediction[0] + 1
        print(gesture_dict[gesture])

        if (gesture != prev_gesture):
            if (gesture == 1):
                keyboard.press(Key.media_volume_down)
                prev_gesture = gesture
            elif (gesture == 2):
                keyboard.press(Key.media_previous)
                prev_gesture = gesture
            elif (gesture == 7):
                keyboard.press(Key.media_play_pause)
                prev_gesture = gesture
            elif (gesture == 4):
                keyboard.press(Key.media_next)
                prev_gesture = gesture
            elif (gesture == 5):
                keyboard.press(Key.media_volume_up)
                prev_gesture = gesture
            elif (gesture == 6):
                prev_gesture = gesture
            elif (gesture == 12):
                print("No Gesture Detected")
                prev_gesture = gesture
            else:
                print("Incorrect Gesture")

    #sleep(0.2)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()