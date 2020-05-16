import cv2
import numpy as np
import keras
from keras.models import load_model
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

model = load_model('model8.h5')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    kernel = np.ones((3,3),np.uint8)

    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100,100),(300,300),(0,255,0),5)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask,kernel,iterations = 4)
    mask = cv2.GaussianBlur(mask,(5,5),100)

    cv2.imshow('Frame',frame)
    cv2.imshow('Mask',mask)

    mask = mask.reshape((200,200,1))
    test_mask = np.expand_dims(mask, axis = 0)

    gesture = model.predict_classes(test_mask)

    g = cv2.waitKey(1) & 0xFF
    if g == ord('k'):
        print(gesture[0])

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()