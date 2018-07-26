#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 01 00:28:22 2018
Cortex creation functions
@author: Piotr Ozimek
"""
import numpy as np
from numpy.linalg import norm
import math
from scipy.spatial.distance import cdist

    
def LRsplit(loc):
    #TODO: Keep note jaggies fix
    """Split the receptive fields into two halves, left and right.
    NOTE: this function needs retinal locs, not a raw tessellation!!
    [x, y, V_index]"""
    left, right = [], []            
    for i in range(len(loc)):
        entry = np.append(loc[i,:2], [i])
        #TODO 2 overlapping halves, ideally overlap is in its own set?
        if loc[i,0] < 0: left.append(entry)
        else: right.append(entry)

    return np.array(left), np.array(right)

""" 
Returns R_loc and L_loc. Each node location is described with:
[x (theta), y(d), node_index, 0, dist_5, node_sigma, node_width]

#Major revisions:
- extended output data structure.
- target_d5 - desired min dist_5. Or mean, not sure which to use.
- node_sigma and node_width inserted in cort_prep.""" 
#TODO fix final jaggies, check for numeric accuracy of gaussian offsets and 
#kernel centres because you cant find a solution to the issue elsewhere
def cort_map(L, R, target_d5=1.0, alpha=15):
    """Given raw left and right hemifields, compute the cortical map """
    #compute cortical coordinates
    #TODO: loop over left and right, or impossible?
    L_loc = np.zeros((len(L),7), dtype='float64')
    L_loc[:,2] = L[:,2]
    L_loc[:,0] = np.arctan2(L[:,1], L[:,0] - alpha) 
    L_loc[:,0] -= np.sign(L_loc[:,0]) * math.pi  #shift theta by pi
    L_loc[:,1] = norm([ L[:,0] - alpha, L[:,1] ], axis=0)
    
    R_loc = np.zeros((len(R),7), dtype='float64')
    R_loc[:,2] = R[:,2]
    R_loc[:,0] = np.arctan2(R[:,1], R[:,0] + alpha)
    R_loc[:,1] = norm([ R[:,0] + alpha, R[:,1] ], axis = 0)
    
    #equate mean distances along x and y axis
    #Note: cdist requires a >=2 dimensional array, hence the messy code
    #TODO: refactor into aux fn
    #x (theta)
    L_theta = np.zeros((len(L_loc),2), dtype='float64')
    R_theta = np.zeros((len(R_loc),2), dtype='float64')
    L_theta[:,0] = L_loc[:,0]
    R_theta[:,0] = R_loc[:,0]
    L_xd = np.mean(cdist(L_theta,L_theta))
    R_xd = np.mean(cdist(R_theta,R_theta))
    xd = (L_xd+R_xd)/2
    
    #y (r)
    L_r = np.zeros((len(L_loc),2), dtype='float64')
    R_r = np.zeros((len(R_loc),2), dtype='float64')
    L_r[:,1] = L_loc[:,1]
    R_r[:,1] = R_loc[:,1]
    L_yd = np.mean(cdist(L_r,L_r))
    R_yd = np.mean(cdist(R_r,R_r))
    yd = (L_yd+R_yd)/2

    #scale
    L_loc[:,0] *= yd/xd
    R_loc[:,0] *= yd/xd
    
    #set dist_5
    ##workload split into chunks due to cdist exhausting RAM
    #TODO: refactor into aux fn
    print "Computing Gaussian interpolation fields..."
    for loc in [L_loc, R_loc]:
        length = len(loc)
        chunk = 5000
        num = length/chunk + np.sign(length%chunk)
        
        dist_5 = np.zeros(length, dtype='float64')
        for j in range(num):
            print "Processing chunk " + str(j)
            s = np.sort(cdist( loc[j*chunk : (j+1)*chunk,:2], loc[:,:2]))
            dist_5[j*chunk:(j+1)*chunk] = np.mean(s[:,1:6], 1)
        
        loc[:,4] = dist_5
    
    #compute mean dist_5 for tighest 5 nodes
    d5 = np.append(L_loc[:,4], R_loc[:,4])
    sort_d5 = np.sort(d5)
    mean_d5 = np.mean(sort_d5[:5])
    
    #impose target mean dist_5 for tighest nodes
    L_loc[:,:2] *= (1.0/mean_d5)*target_d5
    R_loc[:,:2] *= (1.0/mean_d5)*target_d5
    
    #Adjust dist_5 to reflect new scale
    L_loc[:,4] *= (1.0/mean_d5)*target_d5
    R_loc[:,4] *= (1.0/mean_d5)*target_d5
    
    return L_loc, R_loc

def cort_prepare(L_loc, R_loc, shrink=1.0, min_kernel=3, kernel_ratio = 4.0, sigma_base = 1.0):
    #Set sigmas
    L_loc[:,5] = sigma_base * L_loc[:,4]
    R_loc[:,5] = sigma_base * R_loc[:,4]
    
    #too late of an hour for a cleaner solution, blame np.maximum
    #Set kernel widths
    Lk = np.zeros((len(L_loc)))
    Lk[:] = min_kernel
    Rk = np.zeros((len(R_loc)))
    Rk[:] = min_kernel
    L_loc[:,6] = np.maximum(Lk, np.ceil(kernel_ratio*L_loc[:,4]))
    R_loc[:,6] = np.maximum(Rk, np.ceil(kernel_ratio*R_loc[:,4]))
    
    #TODO make all coeff objects not be (1,len) in shape
    L_coeff = np.ndarray((1, len(L_loc)),dtype='object')
    R_coeff = np.ndarray((1, len(R_loc)),dtype='object')
    coeffs = [L_coeff, R_coeff]

    locs = [L_loc, R_loc]    
    for j in range(len(locs)):
        loc = locs[j]
        coeff = coeffs[j]
        
        for i in range(len(loc)):
            k_width = np.uint8(loc[i,6])
            
            #Kernel centre coodinates (odd/even contingent)
            cx, cy = xy_sumitha(loc[i,0], loc[i,1], k_width)
            
            #Obtain subpixel accurate offsets of gaussian from kernel centre
            rx = loc[i][0] - cx
            ry = loc[i][1] - cy
            rloc = np.array([rx, ry])
            
            #Set new coordinates for node kernel centre
            loc[i,0] = cx
            loc[i,1] = cy
            
            #place proper gaussian in coeff[i]
            coeff[0,i] = gausskernel(k_width, rloc, loc[i,5])
            coeff[0,i] /= np.sum(coeff[0,i]) #normalization
    
    ###############
    
    #bring min(x) to 0
    L_loc[:,0] -= np.min(L_loc[:,0])
    R_loc[:,0] -= np.min(R_loc[:,0])
    #flip y and bring min(y) to 0
    L_loc[:,1] = -L_loc[:,1]
    R_loc[:,1] = -R_loc[:,1]
    L_loc[:,1] -= np.min(L_loc[:,1])
    R_loc[:,1] -= np.min(R_loc[:,1])    
    
    #k_max more pixels of space from all sides for kernels to fit
    k_max = max(max(L_loc[:,6]), max(R_loc[:,6]))
    L_loc[:,:2] += k_max
    R_loc[:,:2] += k_max 
    
    #shrinking (avoid)
    L_loc[:,[0,1,4]] *= shrink
    R_loc[:,[0,1,4]] *= shrink
    
    #compute cortical image size
    cort_y = max(R_loc[:,1].max(), L_loc[:,1].max()) + k_max/2 + 1
    cort_x = max(R_loc[:,0].max(), L_loc[:,0].max()) + k_max/2 + 1
    
    cort_size = (int(cort_y),int(cort_x))    
    
    return L_loc, R_loc, L_coeff, R_coeff, cort_size


#TODO rename this fn, explain in a comment
def xy_sumitha(x,y,k_width):
    k_width = int(k_width) #this will change
    
    #if odd size mask -> round coordinates
    if k_width%2 != 0:
        cx = round(x)
        cy = round(y)
        
    #else if even size mask -> 1 decimal point coordinates (always .5)
    else:
        cx = round(x) + np.sign(x-round(x))*0.5
        cy = round(y) + np.sign(y-round(y))*0.5
    
    return cx, cy

#Gauss(sigma,x,y) function, 1D
def gauss(sigma,x,y,mean=0):
    d = np.linalg.norm(np.array([x,y]))
    return np.exp(-(d-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#Kernel(width,loc,sigma,x,y)
def gausskernel(width,loc,sigma):
    #TODO refactor
    #location is passed as np array [x,y]
    k = np.zeros((width, width))    

    #subpixel accurate coords of gaussian centre
    dx = width/2 + np.round(loc[0],decimals=1) - int(loc[0])
    dy = width/2 + np.round(loc[1],decimals=1) - int(loc[1])    
    
    for x in range(width):
        for y in range(width):
            k[y,x] = gauss(sigma,dx-x,dy-y)
    
    return k