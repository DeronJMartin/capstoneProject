import cv2
import numpy as np
import math
from time import sleep
import os
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

#Data Image Directory Definition
directory = '' + os.getcwd() + '/data_images'
os.chdir(directory)
dirlist = sorted_alphanumeric(os.listdir())
dirlist_len = len(dirlist)

#Loop through all the images
for d in range(1):

    #Image Name
    print(dirlist[d])

    #Read Image
    image = cv2.imread(dirlist[d])
    kernel = np.ones((3,3),np.uint8)

    #Region of Interest
    roi = image[100:300, 100:300]

    #Conversion to HSV Color Space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #Skin Filter
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    #Skin Color Extraction
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #Extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask, kernel, iterations = 4)

    #Blur the image to reduce noise
    mask = cv2.GaussianBlur(mask,(5,5),100)

    #Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Find Contour with the Largest Area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    #Approximation of Contour
    epsilon = 0.0005*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    #Convex Hull Algorithm
    hull = cv2.convexHull(cnt)

    #Area of Convex Hull
    areahull = cv2.contourArea(hull)

    #Area of Contour(Hand)
    areacnt = cv2.contourArea(cnt)

    #Area Ration
    #(Total Area - Area of Hand)/Total Area * 100
    arearatio = ((areahull-areacnt)/areacnt)*100

    #Find the Convexity Defects
    hull = cv2.convexHull(approx, returnPoints = False)
    defects = cv2.convexityDefects(approx, hull)

    #Count of Convexity Defects
    l = 0

    print(defects)

    #Finding the total number of Convexity Defects
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        print(str(start)+ ' ' +str(end)+ ' ' +str(far))
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
    
    #Minimum Value of 1 is set    
    l+=1

