#OpenCV - Computer Vision and Image Processing
import cv2
#NumPy - Array and Matrix Handling
import numpy as np
#Kerras - Framework to easily make models
import keras
from keras.models import load_model
#TensirFlow - Backend for Keras
import tensorflow as tf
#Pynput - Allows keypres simulation
import pynput
from pynput.keyboard import Key, Controller
#Sleep - Delay function
from time import sleep

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    e = ''
"""

#Define Gestures
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

#Load Trained Model
model = load_model('model_Final3.h5')

#Keyboard Controller
keyboard = Controller()

#Video Capture Initialization
cap = cv2.VideoCapture(0)

#Flag to store previous gesture no. (Used while processing logic)
prev_gesture = -1

#Total number of mask images captured (30 images capotured per second)
mask_num = -1

while True:
    #Store webcam capture in frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    #Kernel used for image processing
    kernel = np.ones((3,3), dtype=np.uint8)

    #Region of Interest
    #Extract Region of Interest
    roi = frame[100:300, 100:300]
    #Draw square to indicate region of interest
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 5)

    #Convert to HSV Color Space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #Define Skin Color range
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    #Mask processing - all skin colored pixels are converted to white
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, kernel, iterations = 4)
    mask = cv2.GaussianBlur(mask, (5,5), 100)

    #Increment masks captured
    mask_num += 1

    #Display Frame and Mask
    cv2.imshow('Frame',frame)
    cv2.imshow('Mask',mask)

    #Add dimensions to mask for prediction
    mask = mask.reshape((1,200,200,1))

    #Logic to process 1 frame per second
    if (mask_num % 30 == 29):
        prediction = model.predict_classes(mask)
        gesture = prediction[0] + 1
        #print(gesture_dict[gesture])

        if (gesture != prev_gesture):
            #One Finger
            if (gesture == 1):
                keyboard.press(Key.media_volume_down)
                prev_gesture = gesture
                print("Decrease Volume")
            #Two Finger
            elif (gesture == 2):
                keyboard.press(Key.media_previous)
                prev_gesture = gesture
                print("Previous")
            #Front Palm
            elif (gesture == 7):
                keyboard.press(Key.media_play_pause)
                prev_gesture = gesture
                print("Play/Pause")
            #Four Finger
            elif (gesture == 4):
                keyboard.press(Key.media_next)
                prev_gesture = gesture
                print("Next")
            #Five Finger
            elif (gesture == 5):
                keyboard.press(Key.media_volume_up)
                prev_gesture = gesture
                print("Increase Volume")
            #Closed Fist - Neutral Gesture
            elif (gesture == 6):
                prev_gesture = gesture
            #No Gesture
            elif (gesture == 12):
                print("No Gesture Detected")
                prev_gesture = gesture
            else:
                print("Incorrect Gesture")

    #Press 'Esc' to exit the function
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

#Release Webcam
cap.release()
#Destroy Output Windows
cv2.destroyAllWindows()