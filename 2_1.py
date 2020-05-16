import cv2
import numpy as np
import math
import os
import re

#Natural Sorting Function
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

#Video Capture Intialization
cap = cv2.VideoCapture(0)

#Data Image Directory Definition
directory = '' + os.getcwd() + '/data_images'
os.chdir(directory)
dirlist = sorted_alphanumeric(os.listdir())
if not dirlist:
    current_data_image = '0'
else:
    current_data_image = os.path.splitext(dirlist[-1])[0]

while(True):
    #Create Image from Webcam Capture
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    #Create Kernel
    kernel = np.ones((3,3),np.uint8)

    #Region of Interest
    roi = frame[100:300, 100:300]

    #Display for Region of Interest
    cv2.rectangle(frame,(100,100),(300,300),(0,0,255),5)

    #Conversion to HSV Color Space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #Skin Filter
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    #Skin Color Extraction
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #Extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #Blur the image to reduce noise
    mask = cv2.GaussianBlur(mask,(5,5),100)

    #Display Video and Mask
    cv2.imshow('Frame',frame)
    cv2.imshow('Mask',mask)

    #Exit Sequence Wait
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

#Exit Sequence
cap.release()
cv2.destroyAllWindows()
