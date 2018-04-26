#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:24:06 2018

Utility functions used throughout the project

@author: Piotr Ozimek
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import cPickle as pickle

def bgr2rgb(im):
    return np.dstack((im[:,:,2], im[:,:,1], im[:,:,0]))

def camopen():
    cap = 0
    camid = 0
    cap = cv2.VideoCapture(camid)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    
    while not cap.isOpened():
        print str(camid) + ' failed retrying\n'
        cv2.VideoCapture(camid).release()
        cap = cv2.VideoCapture(camid)
        camid += 1  
    return cap

#Run this if cam stops working
def camclose(cap):
    cv2.destroyAllWindows()
    cap.release()
    cv2.VideoCapture(-1).release()
    cv2.VideoCapture(0).release()

#Take a pic with the webcam
def snap():
    cap = camopen()
    ret, img = cap.read()
    camclose()
    return bgr2rgb(img)

def picshow(pic, size=(10,10)):    
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(pic, interpolation='none', cmap='gray')
    plt.show()

def loadPickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)