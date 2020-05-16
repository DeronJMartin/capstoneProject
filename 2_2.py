import cv2
import numpy as np
import os
import re
from time import sleep

"""
List of Gestures
1 - One Finger
2 - Two Fingers
3 - Three Fingers
4 - Four Fingers
5 - Five Fingers
6 - Closed Fist
7 - Front Palm
8 - Side Palm
9 - Shaka
10 - Rock1
11 - Rock2
"""

#Natural Sorting Function
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

#Video Capture Intialization
cap = cv2.VideoCapture(0)

#Index of Gesture to Capture
gestureIndex = 11

#Data Image Directory Definition
directory = '' + os.getcwd() + '/data_images/test'
os.chdir(directory)
directory = '' + os.getcwd() + '/' + sorted_alphanumeric(os.listdir())[gestureIndex-1]
os.chdir(directory)
dirlist = sorted_alphanumeric(os.listdir(directory))
if (not dirlist) or (dirlist[0] == 'raw_data.txt'):
    current_data_image = '0'
else:
    current_data_image = os.path.splitext(dirlist[-2])[0]

#Capture Loop Control Flag
captureFlag = False

#Number of Images to Capture
numCaptures = 1
j = numCaptures

#Text File for Data
textfile = open('raw_data.txt','a')

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

    #Frame Capture Sequence
    k = cv2.waitKey(1) & 0xFF
    if k == ord('l'):
        captureFlag = True
        j=0
    
    while(captureFlag and j<numCaptures):
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

        #Write Mask Image to File
        current_data_image = str(int(current_data_image)+1)
        filename = current_data_image + '.jpeg'
        cv2.imwrite(filename, mask)

        #Loop Control
        j+=1

        #Output number of Images written
        print('Total Images:'+current_data_image+'\tImages Captured: ' + str(j))

        #Write Text Data
        textdata = np.zeros((1,11),dtype=np.uint8)
        textdata[0][gestureIndex-1] = 1
        textstr = str(textdata[0][0])
        for i in range(1,11):
            textstr += ' ' + str(textdata[0][i])
        textfile.writelines(textstr)
        sleep(0.02)

    #Exit Sequence Wait
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

#Exit Sequence
cap.release()
cv2.destroyAllWindows()
textfile.close()
