#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 21:40:41 2018

PhD Cortex - refactoring, object model

@author: Piotr Ozimek
"""

import numpy as np
from cuda_objects import CudaCortex
from retinavision.utils import loadPickleNonbin, project, loadPickle

#TODO: overlapping hemifields?

class Cortex:
    def __init__(self, gpu=True):
        self.hemishape = 0 #matrix size of one cortical half (hemisphere)
        self.N = 0
        self.Rloc = 0
        self.Lloc = 0
        self.Lcoeff = 0
        self.Rcoeff = 0
        
        self._cudaCortex = CudaCortex() if gpu else None
        self._V = 0 
        self.Lnorm = 0 
        self.Rnorm = 0 
        self._Limg = 0
        self._Rimg = 0
        self._image = 0 
        self._loadcount = 0 #re-generate norm imgs only if new file loaded
        self._normid = 0 #id of normalizatio images
        
    def info(self):
        print "Rloc, Lloc - Nx7 arrays defined as follows:\n"\
            "[x (theta), y(d), imagevector_index, 0, dist_5, kernel_sigma, kernel_width]"
       
    def loadLocs(self, leftpath, rightpath):
        self.Rloc = loadPickleNonbin(rightpath)
        self.Lloc = loadPickleNonbin(leftpath)
        self._loadcount += 1
        self.validate()
    
    def loadCoeffs(self, leftpath, rightpath):
        self.Rcoeff = loadPickle(rightpath)
        self.Lcoeff = loadPickle(leftpath)
        self._loadcount += 1
        self.validate()
    
    def validate(self):
        if self.Rloc is not 0 and self.Rcoeff is not 0:
            n1 = len(self.Rloc) + len(self.Lloc)
            n2 = len(self.Rcoeff[0]) + len(self.Lcoeff[0])
            assert(n1 == n2) #invalid coeffs-locs pair check
            self.N = n1
            self.prepare()            
    
    def prepare(self):
        """Compute cortical image shape and normalization images"""
        Rwidth = int(np.abs(self.Rloc[:,0].max() + self.Rloc[:,6].max()/2.0))
        Lwidth = int(np.abs(self.Lloc[:,0].max() + self.Lloc[:,6].max()/2.0))
        Rheight = int(np.abs(self.Rloc[:,1].max() + self.Rloc[:,6].max()/2.0))
        Lheight = int(np.abs(self.Lloc[:,1].max() + self.Lloc[:,6].max()/2.0))

        self.hemishape = (max(Rheight, Lheight), max(Rwidth, Lwidth))
        #re-generate norm imgs only if new file loaded
        if self._normid != self._loadcount: self.cort_norm_img()
    
    def cort_norm_img(self):
        L_norm = np.zeros(self.hemishape, dtype='float64')
        R_norm = np.zeros(self.hemishape, dtype='float64')
        norms = [L_norm, R_norm]
        
        locs = [self.Lloc, self.Rloc]
        coeffs = [self.Lcoeff, self.Rcoeff]
        
        for i in [0,1]:
            nimg = norms[i]
            loc = locs[i]        
            coeff = coeffs[i]       
            n = len(loc)
            
            for i in range(n - 1,-1,-1):
                nimg = project(coeff[0,i], nimg, loc[i,:2][::-1])

        if self._cudaCortex:
            self._cudaCortex.set_cortex(self.Lloc, self.Rloc, self.Lcoeff, \
                self.Rcoeff, L_norm, R_norm, self.hemishape)

        self.Lnorm, self.Rnorm = L_norm, R_norm
        self._normid = self._loadcount
    
    def cort_img(self, V):
        self.validate()
        self._V = V
        rgb = len(V.shape) == 2
        if self._cudaCortex:
            self._cudaCortex.rgb = rgb
            self.Limg = self._cudaCortex.cort_image_left(V)
            self.Rimg = self._cudaCortex.cort_image_right(V)
            self._image = np.concatenate((np.rot90(self.Limg,1), np.rot90(self.Rimg,-1)), axis=1)
            return self._image

        cort_size = (self.hemishape[0], self.hemishape[1])
        Ln, Rn = self.Lnorm, self.Rnorm
        if rgb: 
            cort_size = (self.hemishape[0], self.hemishape[1], 3)
            Ln = np.dstack((self.Lnorm, self.Lnorm, self.Lnorm))
            Rn = np.dstack((self.Rnorm, self.Rnorm, self.Rnorm))
        
        L_img = np.zeros(cort_size, dtype='float64')
        R_img= np.zeros(cort_size, dtype='float64')
        
        imgs = [L_img, R_img]  
        locs = [self.Lloc, self.Rloc]
        coeffs = [self.Lcoeff, self.Rcoeff]
        
        for j in range(len(imgs)):
            img = imgs[j]
            loc = locs[j]        
            coeff = coeffs[j]
            n = len(loc)
            
            for i in range(n-1,-1,-1):
                c = coeff[0, i]
                if rgb: c = np.dstack((c,c,c))
                ni = int(loc[i,2]) #node index
                img = project(c * V[ni], img, loc[i,:2][::-1])
                
        L = np.uint8(np.divide(L_img, Ln))
        R = np.uint8(np.divide(R_img, Rn))
        
        self.Limg, self.Rimg = L, R
        self._image = np.concatenate((np.rot90(L,1), np.rot90(R,-1)), axis=1)
        return self._image
