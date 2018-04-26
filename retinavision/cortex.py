# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:56:13 2017

Cortex 2.0
Switching from constant sigma to a space-variant sigma (in Gaussian interpolation)
@author: Piotr Ozimek
"""
import numpy as np
import math
from scipy.spatial import distance
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

#Gauss(sigma,x,y) function, 1D
def gauss2(sigma,x,y,mean=0):
    d = np.linalg.norm(np.array([x,y]))
    return np.exp(-(d-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#Kernel(width,loc,sigma,x,y)
def gausskernel2(width,loc,sigma):
    w = float(width)
    #location is passed as np array [x,y]
    k = np.zeros((width, width))    
    shift = (w-1)/2.0

    #subpixel accurate coords of gaussian centre
    dx = loc[0] - int(loc[0])
    dy = loc[1] - int(loc[1])    
    
    for x in range(width):
        for y in range(width):
            k[y,x] = gauss(sigma,(x-shift)-dx,(y-shift)-dy)
    
    return k  

############

"""Split the receptive fields into two halves, left and right"""
#TODO 2 overlapping halves, ideally overlap is in its own set?
def LRsplit(rf_loc):
    left = []
    right = []
    
    for i in range(len(rf_loc)):
        entry = np.append(rf_loc[i,:2],[i])
        if rf_loc[i,0] < 0:
            left.append(entry)
        else:
            right.append(entry)

    L = np.zeros(shape=(3,len(left))) #4057
    R = np.zeros(shape=(3,len(right))) #4135  (+78, its fine)
    
    L = np.array(left)
    R = np.array(right)
    
    # [x,y,V_index]
    return L,R

#revised since v1
""" 
Returns R_loc and L_loc. Each node location is described with:
[x (theta), y(d), node_index, 0, dist_5, node_sigma, node_width]

#Major revisions:
- extended output data structure.

- target_d5 - desired min dist_5. Or mean, not sure which to use.

- node_sigma and node_width inserted in cort_prep.

""" 
#TODO fix final jaggies, check for numeric accuracy of gaussian offsets and 
#kernel centres because you cant find a solution to the issue elsewhere
def cort_map(L,R, target_d5=1.0, alpha=15):
    #compute cortical coordinates
    L_r = np.sqrt((L[:,0]-alpha)**2 + L[:,1]**2)
    R_r = np.sqrt((R[:,0]+alpha)**2 + R[:,1]**2)
    L_theta = np.arctan2(L[:,1],L[:,0]-alpha) 
    L_theta = L_theta-np.sign(L_theta)*math.pi  #** shift theta by pi
    R_theta = np.arctan2(R[:,1],R[:,0]+alpha)
    
    L_loc = np.zeros((len(L_theta),7), dtype='float64')
    L_loc[:,:2] = np.array([L_theta,L_r]).transpose()
    L_loc[:,2] = L[:,2]
    
    R_loc = np.zeros((len(R_theta),7), dtype='float64')
    R_loc[:,:2] = np.array([R_theta,R_r]).transpose()
    R_loc[:,2] = R[:,2]
    
    ##equate mean distances along x and y axis
    #warning - reusing variable names 
    #Note: cdist requires a >=2 dimensional array, hence the messy code
    
    #x (theta)
    L_theta = np.zeros((len(L_loc),2), dtype='float64')
    R_theta = np.zeros((len(R_loc),2), dtype='float64')
    L_theta[:,0] = L_loc[:,0]
    R_theta[:,0] = R_loc[:,0]
    L_xd = np.mean(distance.cdist(L_theta,L_theta))
    R_xd = np.mean(distance.cdist(R_theta,R_theta))
    xd = (L_xd+R_xd)/2
    
    #y (r)
    L_r = np.zeros((len(L_loc),2), dtype='float64')
    R_r = np.zeros((len(R_loc),2), dtype='float64')
    L_r[:,1] = L_loc[:,1]
    R_r[:,1] = R_loc[:,1]
    L_yd = np.mean(distance.cdist(L_r,L_r))
    R_yd = np.mean(distance.cdist(R_r,R_r))
    yd = (L_yd+R_yd)/2
    
    #scale
    L_loc[:,0] *= yd/xd
    R_loc[:,0] *= yd/xd
        
    #set dist_5
    ##workload split into chunks due to cdist exhausting RAM
    t = [L_loc[:,:2], R_loc[:,:2]] #'Tessellations'
    locs = [L_loc, R_loc]
    for i in range(len(t)):
        print "Computing Gaussian interpolation fields " + str(i+1) + "/2..."
        tessellation = t[i]
        length = len(tessellation)
        chunk = 5000
        num = length/chunk
        if length%chunk != 0:
            num += 1
        
        dist_5 = np.zeros(length, dtype='float64')
        for j in range(num):
            print "Processing chunk " + str(j)
            d = distance.cdist(tessellation[j*chunk:(j+1)*chunk], tessellation)
            s = np.sort(d)
            dist_5[j*chunk:(j+1)*chunk] = np.mean(s[:,1:6], 1)
        
        #Insert dist_5 to [4]
        locs[i][:,4] = dist_5        
        
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


"""
Fill in node_sigmas and node_widths.
Create L_coeff and R_coeff
"""
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
  
  
def cort_norm_img(L_loc, R_loc, L_coeff, R_coeff, cort_size):
    L_norm = np.zeros(cort_size, dtype='float64')
    R_norm = np.zeros(cort_size, dtype='float64')
    norms = [L_norm, R_norm]
    
    locs = [L_loc, R_loc]
    coeffs = [L_coeff, R_coeff]
    
    for i in range(len(norms)):
        norm = norms[i]
        loc = locs[i]        
        coeff = coeffs[i]       
        s = len(loc)
        
        for i in range(s-1,-1,-1):
            w = loc[i,6]
            y1 = int(loc[i,1] - w/2+0.5)
            y2 = int(loc[i,1] + w/2+0.5)
            x1 = int(loc[i,0] - w/2+0.5)
            x2 = int(loc[i,0] + w/2+0.5)

            c = coeff[0, i]
            norm[y1:y2,x1:x2] += c
    
    return L_norm, R_norm
       
    
def cort_img(V, L_loc, R_loc, L_coeff, R_coeff, Ln, Rn, cort_size):
    rgb = len(V.shape) == 2
    if rgb: 
        cort_size = (cort_size[0], cort_size[1], 3)
        Ln = np.dstack((Ln, Ln, Ln))
        Rn = np.dstack((Rn, Rn, Rn))
    
    L_img = np.zeros(cort_size, dtype='float64')
    R_img= np.zeros(cort_size, dtype='float64')
    
    imgs = [L_img, R_img]  
    locs = [L_loc, R_loc]
    coeffs = [L_coeff, R_coeff]
    
    for j in range(len(imgs)):
        img = imgs[j]
        loc = locs[j]        
        coeff = coeffs[j]
        s = len(loc)
        
        for i in range(s-1,-1,-1):
            w = loc[i,6]
            y1 = int(loc[i,1] - w/2+0.5)
            y2 = int(loc[i,1] + w/2+0.5)
            x1 = int(loc[i,0] - w/2+0.5)
            x2 = int(loc[i,0] + w/2+0.5)

            ni = int(loc[i,2]) #node index
            c = coeff[0, i]
            if rgb: add = np.dstack((c,c,c)) * V[ni]
            else: add = c*V[ni]
            img[y1:y2,x1:x2] += add
            
    L_img = np.uint8(np.divide(L_img,Ln))
    R_img = np.uint8(np.divide(R_img,Rn))
        
    
    return L_img, R_img

def show_cortex(left,right):
    L = np.rot90(left,1)
    R = np.rot90(right,-1)
    LRfig = plt.figure(figsize=(10,5))
    LRgrid = ImageGrid(LRfig, 111, nrows_ncols=(1,2))
    LRgrid[0].imshow(L, interpolation='none', cmap='gray')
    LRgrid[0].axis('off')
    LRgrid[1].imshow(R, interpolation='none', cmap='gray')
    LRgrid[1].axis('off')
    plt.show()


#Gauss(sigma,x,y) function, 1D
def gauss(sigma,x,y,mean=0):
    d = np.linalg.norm(np.array([x,y]))
    return np.exp(-(d-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#Kernel(width,loc,sigma,x,y)
def gausskernel(width,loc,sigma):
    #location is passed as np array [x,y]
    k = np.zeros((width, width))    

    #subpixel accurate coords of gaussian centre
    dx = width/2 + np.round(loc[0],decimals=1) - int(loc[0])
    dy = width/2 + np.round(loc[1],decimals=1) - int(loc[1])    
    
    for x in range(width):
        for y in range(width):
            k[y,x] = gauss(sigma,dx-x,dy-y)
    
    return k

#legacy function
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