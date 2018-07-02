#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 26/04/2018

Barebones camera demo showcasing the use of the newest retina codes

@author: Piotr Ozimek
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from retinavision.retina import Retina
from retinavision.cortex import Cortex
from retinavision import datadir, utils
import cPickle as pickle
from os.path import join


#Open webcam
cap = utils.camopen() #cap is the capture object (global)
ret, campic = cap.read()

#Create and load retina
R = Retina()
R.info()
R.loadLoc(join(datadir, "retinas", "ret50k_loc.pkl"))
R.loadCoeff(join(datadir, "retinas", "ret50k_coeff.pkl"))

#Prepare retina
x = campic.shape[1]/2
y = campic.shape[0]/2
fixation = (y,x)
R.prepare(campic.shape, fixation)

#Create and prepare cortex
C = Cortex()
lp = join(datadir, "cortices", "Ll.pkl")
rp = join(datadir, "cortices", "Rl.pkl")
C.loadLocs(lp, rp)
C.loadCoeffs(join(datadir, "cortices", "Lcoeff.pkl"), join(datadir, "cortices", "Rcoeff.pkl"))

while True:
    ret, img = cap.read()
    if ret is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        V = R.sample(img, fixation)
        #tight = R.backproject_tight_last()
        tight = R.backproject_last()
        cimg = C.cort_img(V)
        
        cv2.namedWindow("inverted", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("inverted", tight) 
        
        cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("input", img) 
        
        cv2.namedWindow("cortical", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("cortical", cimg) 
        
        key = cv2.waitKey(10)
        
        if key == 43: #+
            print ''
#        elif key == 45: #-
#            print ''
#        elif key == 97: #a
#            print 'switching autoscaling...'
#            cv2.destroyAllWindows()
#            autoscale = not autoscale
#        elif key == 105: #'i
#            cv2.destroyWindow("inverted")
#            showInverse = not showInverse            
#        elif key == 99: #c
#            showCortex = not showCortex
#            cv2.destroyWindow("cortex")
#        elif key == 119: #w
#            imShrink += 1
#            imShrink = min(6, imShrink)
#            prep()
#        elif key == 115: #s
#            imShrink -= 1
#            imShrink = max(imShrink, 1)
#            prep()
        elif key == 27: #esc
            break
#        elif key != -1:
#            print key
            
utils.camclose(cap)
