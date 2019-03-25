#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:10:08 2019

Pyramid webcam demo.
@author: Piotr Ozimek

TODO:
    - refactoring Pyramid.py
    - Pytorch/accelerate the pyramidal sampling and backproj functions
    - Make a cpu version of pyramid fns (backproject, visualize)
    - Make DoG work on GPU proper (read NOTES)
    
NOTES:
    gpu_pyr_debug.py contains the latest struggle with getting DoG to work on
    the GPU. Unfortunately the subtraction of the 2 pyramidal imagevectors 
    from each other causes the resultant imagevectors to be too small for the 
    GPU codes to normalize on the device (quantization + unhandled negatives).
    This is why in the code below the function visualize() handles normalization
    for the DoG images. In the future ask Lorinc to: Dockerize/package the
    development environment for retinavision-gpu and to have the codes return
    floats if a flag is passed. Actually, check if he didn't do that already 
    seeing how unnormalized backprojections return as floats.
    
    For now: GPU accelerated Gaussian pyramids work fine, but DoG pyramids
    need to be normalized on the CPU and require different function calls.
"""

import cv2
import numpy as np
from retinavision.retina import Retina
from retinavision.cortex import Cortex
from retinavision import datadir, utils
from retinavision.pyramid import Pyramid
from os.path import join

def visualize(BI, title, pyr, dog=False):
    for i in range(len(BI[:-1])):
        if dog: im = np.true_divide(BI[i], pyr.norm_maps[i])
        else: im = BI[i]
        cv2.namedWindow(title+str(i), cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title+str(i), im)

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

#Prepare pyramid
pyr_path = join(datadir, "pyramid")
L = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_tessellations.pkl"))
L2 = utils.loadPickle(join(pyr_path, "50K_pyr_wide_tessellations.pkl"))
N = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_normmaps.pkl"))
N2 = utils.loadPickle(join(pyr_path, "50K_pyr_wide_normmaps.pkl"))
C = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_coeffs.pkl"))
C2 = utils.loadPickle(join(pyr_path, "50K_pyr_wide_coeffs.pkl"))

narrow = Pyramid(tess = L, coeffs = C, N=N, R=R)
wide = Pyramid(tess = L2, coeffs = C2, N=N2, R=R)

donarrow = False
dowide = False
dodog = True

while True:
    ret, img = cap.read()
    if ret is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        V = R.sample(img, fixation)
        tight = R.backproject_tight_last()
        
        narrow_PV = narrow.sample(V)
        wide_PV = wide.sample(V)
        DoG = wide_PV - narrow_PV

        cv2.namedWindow("inverted", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("inverted", tight) 
        
        cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("input", img)
        
        
        if donarrow: narrow_vis = narrow.backproject_last()
        if dowide: wide_vis = wide.backproject_last()
        if dodog: DoG_vis = narrow.backproject(DoG, R._imsize, fixation, n=False)
        
        if donarrow: visualize(narrow_vis, "Narrow", narrow, dog=False)
        if dowide: visualize(wide_vis, "Wide", wide, dog=False)
        if dodog: visualize(DoG_vis, "DoG", narrow, dog=True)
                
        key = cv2.waitKey(10)
        if key == 27: #esc
            break
        elif key == 49: #1
            cv2.destroyWindow("Narrow0")
            cv2.destroyWindow("Narrow1")
            cv2.destroyWindow("Narrow2")
            donarrow = not donarrow
        elif key == 50: #2
            cv2.destroyWindow("Wide1")
            cv2.destroyWindow("Wide2")
            cv2.destroyWindow("Wide0")
            dowide = not dowide 
        elif key == 51: #3
            cv2.destroyWindow("DoG1")
            cv2.destroyWindow("DoG2")
            cv2.destroyWindow("DoG0")
            dodog = not dodog
        elif key != -1:
            print(key)

utils.camclose(cap)