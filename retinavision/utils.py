#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:24:06 2018

Utility functions used throughout the project

TODO: create a separate tessellation utils file

@author: Piotr Ozimek
"""
import sys
py = sys.version_info.major

import matplotlib.pyplot as plt
import cv2
import numpy as np


if py == 2: import cPickle as pickle
elif py == 3: 
    import torch
    import pickle


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
def project(source, target, location, v=False):
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
    
    try: target[y1:y2, x1:x2] += source[s_y1:s_y2, s_x1:s_x2]
    except Exception as E:
        print(y1, y2, x1, x2)
        print(s_y1, s_y2, s_x1, s_x2)
        print(source.shape)
        print(target.shape)
        print(location)
        raise E
    
    if v:
        print(y1, y2, x1, x2)
        print(s_y1, s_y2, s_x1, s_x2)
        print(source.shape)
        print(target.shape)
        print(location)
    
    return target


""" Convenience functions""" 
def bgr2rgb(im):
    return np.dstack((im[:,:,2], im[:,:,1], im[:,:,0]))

#OH BOY PYTHON 3 SURELY HURTS
def normal_round(n):
    if n - np.floor(np.abs(n)) < 0.5:
        return np.floor(n)
    return np.ceil(n)

#i = int, r = round.
def ir(val):
    return int(normal_round(val))
       
#Unused 
def normalize(V):
    return V *(255/V.max())

#Get GPU memory stats
def memshow():
    total = torch.cuda.get_device_properties(0).total_memory
    a = torch.cuda.memory_allocated()
    c = torch.cuda.memory_cached()
    r = total-torch.cuda.memory_allocated()
    print("Allocated:", a)
    print("Cached:", c)
    print("Remaining:", r, "(", r/total,")")

""" Pickling functions. Pain ahead. 
Py36 Pickle is not compatible with Py2 cPickle, so latin1 encoding is needed."""
def loadPickle(path):
    with open(path, 'rb') as handle:
        if py == 3: return pickle.load(handle, encoding='latin1')
        return pickle.load(handle)
    
def writePickle(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)

#Use this function to open pickled files in py2 that wont open otherwise
#The associated bug usually mentions a missing module.        
def loadPickleNonbin(path):
    with open(path, 'r') as handle:
        if py == 3: return pickle.load(handle, encoding='latin1')
        return pickle.load(handle)        

"""Camera and visualisation functions""" 
def camopen():
    cap = 0
    camid = 0
    cap = cv2.VideoCapture(camid)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    
    while not cap.isOpened():
        print(str(camid) + ' failed, retrying\n')
        cv2.VideoCapture(camid).release()
        cap = cv2.VideoCapture(camid)
        camid += 1  
    return cap

#Run this if cam stops working. Does not work in py3 (cam never closes?)
def camclose(cap):
    cap.release()
    cv2.destroyAllWindows()

#Take a pic with the webcam, return RGB!
def snap():
    cap = camopen()
    ret, img = cap.read()
    camclose(cap)
    return bgr2rgb(img)

def picshow(pic, size=(10,10), cmap = 'gray'):    
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(pic, interpolation='none', cmap=cmap)
    plt.show()
    
def imshow(pic, size=(10,10)):    
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(pic, interpolation='none')
    plt.show()
    
def scatter(X,Y, size=(10,10)):    
    plt.figure(figsize=size)
    plt.scatter(X, Y)
    plt.show()
    
"""Tessellation specific functions"""
def tessplot(T, s = (10,10), c = None, cmap=None, size=1, axis='on', marker='o'):    
    plt.figure(figsize = s)
    if axis=='off': plt.axis('off')
    plt.scatter(T[:,0], T[:,1], s = size, c = c, cmap=cmap, marker=marker)
    plt.show()
    
    
"""GPU accelerated cdist. y = rows, x = columns. Demons ahead."""
def cdist_torch(x, y):
    xs, ys = x.shape[0], y.shape[0]
    dims = x.shape[1] if len(x.shape) == 2 else 0
    if dims: assert(x.shape[1] == y.shape[1])
    
    #Pre-compute required GPU memory. Assuming float64
    batch_process = False
    x_mem, y_mem, xd_mem, dist_mem = 8, 8, 8, 8
    for i in x.shape: x_mem *= i
    for i in y.shape: y_mem *= i
    for i in ys, xs: dist_mem *= i
    if dims:
        for i in ys, xs, dims: xd_mem *= i
    else:
        xd_mem = dist_mem
    mem = dist_mem + xd_mem #peak memory usage
    
    total = torch.cuda.get_device_properties(0).total_memory - 2000 
    if mem > total: 
        batch_process = True
        assert(total - (x_mem + y_mem) > 0), "Inputs too large to fit on GPU"
        print("cdist_torch: Insufficient GPU memory, computing in batches...")
        xd_space = total - (x_mem + y_mem)
        #print("Space for xd = ", xd_space,", xd size = ", xd_mem)
        
        
    #Start processing cdist
    Y = torch.from_numpy(y).to("cuda")
    X = torch.from_numpy(x).to("cuda")
    
    if batch_process:
        xd_bsize = xd_space // (8 * xs * dims) #size of a batch in rows or columns
        xd_batches = int(np.ceil(xd_mem/xd_space)) #number of batches
        Xd = torch.zeros(ys, xs, dims, device="cpu")
        
        for B in range(xd_batches):
            xd1 = B * xd_bsize
            xd2 = min((B+1) * xd_bsize, ys)
            xd_size = xd2-xd1
            
            if dims: Xdbatch = torch.zeros(xd_size, xs, dims, device="cuda")
            else: Xdbatch = torch.zeros(xd_size, xs, device="cuda")
              
            for i in range(xd1,xd2):
                Xdbatch[i-xd1] = X - Y[i]
            
            Xd[xd1:xd2] = Xdbatch
            del(Xdbatch)
        del X, Y
        dist_bsize = total // ((8 * xs)+ (8 * xs * dims))
        #gotta hold a row of dist and Xd simultaneously
        dist_batches = np.ceil((dist_mem+xd_mem)/total)
        dist = torch.zeros(ys, xs, device="cpu")
        
        for B in range(int(dist_batches)):
            dist1 = B * dist_bsize
            dist2 = min((B+1) * dist_bsize, ys)
            Xd_batch = Xd[dist1:dist2].to("cuda")
            
            if dims: d = torch.norm(Xd_batch,2, dims)
            else: d = np.abs(Xd) #Can't be bothered doing one more dims check
            
            dist[dist1:dist2] = d
            del d, Xd_batch
        out = dist
            
    else: #nice and easy, all in one gulp
        if dims: Xd = torch.zeros(ys,xs,dims,device="cuda")
        else: Xd = torch.zeros(ys,xs,device="cuda")
            
        for i in range(ys):
            Xd[i] = X - Y[i]       
        del X, Y
        
        if dims: dist = torch.norm(Xd,2, dims) #OOM here
        else: dist = np.abs(Xd)
        del Xd
        out = dist.to("cpu")
    
    torch.cuda.empty_cache()
    return out