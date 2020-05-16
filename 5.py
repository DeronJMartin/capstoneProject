import cv2
import numpy as np
import math
import pynput
from pynput.keyboard import Key, Controller
from time import sleep
import os
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

#To access Keyboard Controller
keyboard = Controller()

#VideoCapture Intialization
cap = cv2.VideoCapture(0)

#Outside the Scope of the Loop
gesture = -1

#Data Image Directory Definition
directory = '' + os.getcwd() + '/data_images'
os.chdir(directory)
dirlist = sorted_alphanumeric(os.listdir(directory))
if not dirlist:
    current_data_image = '0'
else:
    current_data_image = os.path.splitext(dirlist[-1])[0]

while(1):
    
    try:
        #Create Capture from Webcam
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        original_frame = frame.copy()
        kernel = np.ones((3,3),np.uint8)

        #Display Video
        cv2.imshow('frame',frame)

        #Region of Interest
        roi = frame[100:300, 100:300]

        #Display for Region of Interest
        cv2.rectangle(frame,(100,100),(300,300),(0,0,255),0)

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
        
        #Find Contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
        #Find Contour with the Largest Area
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
        #Approximation of Contour
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        
        #Convex Hull Algorithm
        hull = cv2.convexHull(cnt)
        
        #Area of Convex Hull
        areahull = cv2.contourArea(hull)

        #Area of Contour(Hand)
        areacnt = cv2.contourArea(cnt)
      
        #Area Ratio
        #(Total Area - Area of Hand)/Total Area * 100
        arearatio=((areahull-areacnt)/areacnt)*100
    
        #Find the Convexity Defects
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        #Count of Convexity Defects
        l=0
        
        #Finding the total number of Convexity Defects
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100,180)
            
            
            #Find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #Distance b/w point and convex hull
            d=(2*ar)/a
            
            #Cosine Rule
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            #Ignore angles > 90 and Ignore points very close to Convex Hull
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], 3)
            
            #Draw lines around Hand
            cv2.line(roi,start, end, [0,0,255], 2)
            
        #Minimum Value of 1 is set    
        l+=1

        #Stores the Previous Gesture, -1 by Default
        prev_gesture = gesture
        
        #Gesture Detection
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                #Nothing in Frame
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<12:
                    #Closed Fist
                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else:
                    #One Finger
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==2:
            #Two Fingers
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
            #Three Fingers
            cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==4:
            #Four Fingers
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==5:
            #Five Fingers
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
            #Error
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        #Display Video and Mask
        cv2.imshow('Frame',frame)
        cv2.imshow('Mask',mask)
                
    except:
        pass
    
    #Frame Capture Sequence
    k = cv2.waitKey(0) & 0xFF
    if k == ord('l'):
        current_data_image = str(int(current_data_image)+1)
        filename = current_data_image + '.jpg'
        cv2.imwrite(filename, original_frame)

    #Exit Sequence
    #Press 'Esc' to Exit
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    
