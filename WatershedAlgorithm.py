# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:52:53 2019

@author: Pranav
"""
#Import modules
import numpy as np
import cv2
from matplotlib import pyplot as plt

#Reading image using cv2 libraries
#img = cv2.imread('coins.png')
img = cv2.imread('a1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# Noise Removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# Background area 
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

#Applying watershed algorithm and marking the regions segmented
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [0,255,255]

#Displaying the segmented image
imS = cv2.resize(img, (612, 368))
cv2.imshow('Segmented Result', imS)
cv2.waitKey()
