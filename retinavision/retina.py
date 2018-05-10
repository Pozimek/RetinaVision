#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:50:12 2018

PhD retina. Object model and code cleanup.

@author: Piotr Ozimek
"""

import numpy as np
from utils import pad, loadPickle, project

#TODO: do something about coeff being (1,X) in shape instead of (X)...
class Retina:
    def __init__(self):
        self.loc = 0
        self.N = 0
        self.coeff = 0
        self.width = 0
        
        self._fixation = 0 #YX tuple
        self._imsize = 0
        self._gaussNorm = 0 #image
        self._gaussNormTight = 0 #image
        self._normFixation = 0 #YX tuple
        self._V = 0
        self._backproj = 0 #image
        self._backprojTight = 0 #image
        
    def info(self):
        print "loc - an Nx7 array containing retinal nodes defined as follows:\n\
    [x, y, d, angle (radians), dist_5, rf_sigma, rf_width]\n\
coeff - an array of variable size gaussian receptive field kernels\n\
V - the imagevector, output of retinal sampling\n\
gaussNorm - Gaussian normalization image for producing backprojections\n\
\n\
REMEMBER: all coordinates are tuples in the Y,X order, not X,Y.\n\
The only exception is the loc array\n\
REMEMBER2: coeff is redundantly wrapped in another matrix for backwards compatibility" 
    
    def loadLoc(self, path):
        self.loc = loadPickle(path)
        self.N = len(self.loc)
        self.width = 2*int(np.abs(self.loc[:,:2]).max() + self.loc[:,6].max()/2.0)
        
    def loadCoeff(self, path):
        self.coeff = loadPickle(path)
       
    def validate(self):
        assert(len(self.loc) == len(self.coeff[0]))
        if self._gaussNormTight is 0: self._normTight()
        
    def _normTight(self): 
        """Produce a tight-fitted Gaussian normalization image (width x width)"""
        GI = np.zeros((self.width, self.width))         
        r = self.width/2.0
        for i in range(self.N - 1, -1, -1): 
            GI = project(self.coeff[0,i], GI, self.loc[i,:2][::-1] + r)
        
        self._gaussNormTight = GI
        
    def prepare(self, shape, fix):
        """Pre-compute fixation specific Gaussian normalization image """
        self.validate()
        self._normFixation = fix
        
        GI = np.zeros(shape[:2])
        GI = project(self._gaussNormTight, GI, fix)
        self._gaussNorm = GI
    
    def sample(self, image, fix):
        """Sample an image"""
        self.validate()
        self._fixation = fix
        self._imsize = image.shape
        rgb = len(image.shape) == 3 and image.shape[-1] == 3
        p = self.width
        pic = pad(image, p, True) #TODO: is this padding faster than calculating sampling windows?
        
        X = self.loc[:,0] + fix[1] + p
        Y = self.loc[:,1] + fix[0] + p
        
        if rgb: V = np.zeros((self.N,3))
        else: V = np.zeros((self.N))
        
        for i in range(0,self.N):
            w = self.loc[i,6]
            y1 = int(Y[i] - w/2+0.5)
            y2 = int(Y[i] + w/2+0.5)
            x1 = int(X[i] - w/2+0.5)
            x2 = int(X[i] + w/2+0.5)
            extract = pic[y1:y2,x1:x2]
            
            c = self.coeff[0, i]
            if rgb: kernel = np.dstack((c,c,c))
            else: kernel = c
            
            m = np.where(np.isnan(extract), 0, 1.0) #mask
            
            if rgb: f = 1.0/np.sum(m*kernel, axis = (0,1)) #TODO fix invalid value warnings
            else: f = 1.0/np.sum(m*kernel)
            
            extract = np.nan_to_num(extract)
            if rgb: V[i] = np.sum(extract*kernel, axis=(0,1)) * f
            else: V[i] = np.sum(extract*kernel) * f
       
        self._V = V
        return V
    
    def backproject_last(self):
        return self.backproject(self._V, self._imsize, self._fixation)
    
    def backproject(self, V, shape, fix):
        """Backproject the image vector onto a blank matrix equal in size to
         the input image"""
        self.validate()
        if fix != self._normFixation or shape[:2] != self._gaussNorm.shape:
            self.prepare(shape, fix)
        
        rgb = len(shape) == 3 and shape[-1] == 3
        m = shape[:2]
        
        if rgb: I1 = np.zeros((m[0], m[1], 3))
        else: I1 = np.zeros(m)
        w = self.width
        I1 = pad(I1, w, False)        
        I = np.zeros(m)
        
        for i in range(self.N-1,-1,-1):    
            c = self.coeff[0, i]
            if rgb: c = np.dstack((c,c,c))
            
            I1 = project(c*V[i], I1, self.loc[i,:2][::-1] + fix + w)
        
        I1 = I1[w:-w, w:-w]
        GI = self._gaussNorm
        if rgb: GI = np.dstack((GI,GI,GI))
        I = np.uint8(np.divide(I1,GI)) 
        
        self._backproj = I
        return I
    
    def backproject_tight_last(self):
        return self.backproject_tight(self._V, self._imsize, self._fixation)
    
    def backproject_tight(self, V, shape, fix):
        """Produce a tight-fitted backprojection (width x width, lens only)"""
        #TODO: look at the weird artifacts at edges when the lens is too big for the frame. Might be a small bug
        self.validate()
        rgb = len(shape) == 3 and shape[-1] == 3
        m = self.width
        r = m/2.0
        
        if rgb: I1 = np.zeros((m, m, 3))
        else: I1 = np.zeros((m, m))
        I = np.zeros_like(I1)
        
        for i in range(self.N - 1,-1,-1):
            c = self.coeff[0, i]
            if rgb: c = np.dstack((c,c,c))
            
            I1 = project(c*V[i], I1, self.loc[i,:2][::-1] + r)
    
        GI = self._gaussNormTight
        if rgb: GI = np.dstack((GI,GI,GI)) #TODO: fix invalid value warnings
        I = np.uint8(np.divide(I1,GI)) 
        
        self._backprojTight = I
        return I
    
    #TODO: add the crop function. Tightly crop original image using retinal lens