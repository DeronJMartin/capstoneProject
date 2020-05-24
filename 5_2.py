import cv2
import numpy as np
import pynput
from pynput.keyboard import Key, Controller
import os
import json

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

    os.chdir('live')
    cv2.imwrite('live_mask.jpeg', mask)

    #Gesture Read from JSON
    gesture_file = open('live_gesture.txt')
    gesture = int(gesture_file.read())
    os.chdir('..')

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

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()