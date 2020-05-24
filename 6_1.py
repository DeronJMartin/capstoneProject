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

cap = cv2.VideoCapture(0)

prev_gesture = -1

"""
Password Modes:
1 - Password Set
2 - Password Verification
"""

mask_num = -1

model = load_model('model13.h5')

password_mode = 2
password_len = 0
password_len_c = 0

start_flag = False
end_flag = False

password = []
temp_password1 = []
temp_password2 = []

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

        if (gesture == 6 ):
            start_flag = True

        #Password Handling
        if (password_mode == 1) and (start_flag == True):
            if (password_len < 8):
                if (password_len == 0):
                    print("Password Set\nUse Closed Fist to seperate gestures\n8 Gestures per password")
                if (gesture != prev_gesture):
                    if (gesture != 6) and (gesture != 12):
                        temp_password1.append(gesture)
                        password_len += 1
                        print(gesture_dict[gesture])
                        prev_gesture = gesture

            if (password_len_c < 8) and (password_len == 8):
                if (password_len_c == 0):
                    print("Password Confirm\nUse Closed Fist to seperate gestures\n8 Gestures per password")
                    sleep(2)
                if (gesture != prev_gesture):
                    if (gesture != 6) and (gesture != 12):
                        temp_password2.append(gesture)
                        password_len_c += 1
                        print(gesture_dict[gesture])
                        prev_gesture = gesture

                if (password_len_c == 8) and (password_len == 8):
                    if (temp_password1 == temp_password2):
                        password = temp_password1.copy()
                        password_file = open('password.bin', 'wb')
                        password_bin = bytearray(password)
                        password_file.write(password_bin)
                        password_file.close()
                        print("Password Set!")
                        end_flag = True
                    else:
                        print("Passwords do not match!")
                        end_flag = True

        elif(password_mode == 2) and (start_flag == True):
            password_file = open('password.bin', 'rb')
            password = list(password_file.read())
            if (password_len < 8):
                if (gesture != prev_gesture):
                    if (gesture == 6):
                        prev_gesture = gesture                
                    elif (gesture != 12):
                        temp_password1.append(gesture)
                        password_len += 1
                        print(gesture_dict[gesture])
                        prev_gesture = gesture
            else:
                if (password == temp_password1):
                    print("Password Verified")
                    end_flag = True
                else:
                    print("Incorrect Password Entered. Please Try Again")
                    end_flag = True
    
    #sleep(0.4)

    if (end_flag == True):
        break

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()