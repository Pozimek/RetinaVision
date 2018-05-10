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


"""Matrix manipulation functions""" 
#Pad an image (with 0s or NaNs) from all sides
def pad(img, padding, nans=False):
    size = ()
    for i in range(len(img.shape)):
        if i != 2: size += (img.shape[i] + 2*padding,)
        else: size += (img.shape[i],)

    out = np.zeros(size, dtype = img.dtype)
    
    if nans: out = np.full(size, np.nan)
    out[padding:-padding, padding:-padding] = img
    
    return out

#Project the source image onto the target image at the given location
def project(source, target, location):
    sh, sw = source.shape[:2]
    th, tw = target.shape[:2]
    
    #target frame
    y1 = max(0, ir(location[0] - sh/2.0))
    y2 = min(th, ir(location[0] + sh/2.0))
    x1 = max(0, ir(location[1] - sw/2.0))
    x2 = min(tw, ir(location[1] + sw/2.0))
    
    #source frame
    s_y1 = - ir(min(0, location[0] - sh/2.0 + 0.5))
    s_y2 = s_y1 + (y2 - y1)
    s_x1 = - ir(min(0, location[1] - sw/2.0 + 0.5))
    s_x2 = s_x1 + (x2 - x1)
    
    target[y1:y2, x1:x2] += source[s_y1:s_y2, s_x1:s_x2]
    
    return target

""" Convenience functions""" 
def bgr2rgb(im):
    return np.dstack((im[:,:,2], im[:,:,1], im[:,:,0]))

def ir(val):
    return int(round(val))

def loadPickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)
    
def loadPickleNonbin(path):
    with open(path, 'r') as handle:
        return pickle.load(handle)
    
def writePickle(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)

"""Camera and visualisation functions""" 
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
    camclose(cap)
    return bgr2rgb(img)

def picshow(pic, size=(10,10)):    
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(pic, interpolation='none', cmap='gray')
    plt.show()
    
def scatter(X,Y, size=(10,10)):    
    plt.figure(figsize=size)
    plt.scatter(X, Y)
    plt.show()