#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 6 16:20:09 2018

Retina Pyramids
@author: Piotr Ozimek
"""
from os.path import join
from retinavision.retina import Retina
from retinavision import datadir, utils
import cv2
import numpy as np
from copy import deepcopy

class Pyramid:  
    #TODO: convert Coefficients into numpy object arrays!
    def __init__(self, tess=0, coeffs=0, N=0, R=0):
        self.tessellations = tess
        self.coefficients = coeffs
        self.norm_maps = N
        self.retina = R
        self.PV = 0
        
        #descriptive vars
        self.levels = 0
        self.sigma_factor = 0
        self.rf_fov = 0
        
    """Load pickled *array of tessellations* """
    def loadTess(self, path):
        self.tessellations = utils.loadPickle(path)
        self.levels = len(self.tessellations)
    
    """Load pickled *array of coefficient arrays* """    
    def loadCoeffs(self, path):
        self.coefficients = utils.loadPickle(path)
        #validate
        
    def loadNormMaps(self, path):
        self.norm_maps = utils.loadPickle(path)
        
    def setRetina(self, R):
        self.retina = R
        R.validate()
    
    def info(self):
        print("Tessellations, coefficients and normalization maps are all\n\
              stored as arrays of np.arrays. Starting from Level 0 (retina)\n\
              for the tessellations and from L01 for the coefficients and \n\
              norm_maps variables. This means that the retinal coefficients\n\
              and the retina normalization image are only stored inside the retina variable.")
    
    """Core sampling/pyramidizing function"""
    def sample(self, img, fixation, P):
        self.retina.prepare(img.shape, fixation)
        V0 = self.retina.sample(img, fixation)
        
        PV = np.ndarray(levels, dtype='object')
        PV[0] = V0
        
        for level in range(1,levels):
            C = self.coefficients[level-1]
            #nans signify a node that is outside of the image frame
            nans = np.where(np.isnan(PV[level-1]))
            V = np.zeros(len(C))
            for node in range(len(C)):
                #Omit nans or they'll spill into image centre
                not_nan = np.where(np.logical_not(np.isin(C[node][0],nans)))
                coeff_i = C[node][0][not_nan]
                coeffs = C[node][1][not_nan]
                
                V[node] = np.sum(PV[level-1][coeff_i] * coeffs)
            PV[level] = V
        
        self.PV = PV
        return PV
    
    def backproject_last(self):
        out = self.Gpyramid_backproject(self.PV, self.retina._imsize, self.retina._fixation)
        return out    
    
    def backproject(self, PV, shape, fixation):
        BV = []
        BV.append(PV[0])
        for i in [1,2,3]: 
            V_stream = [PV[i]]
            for j in range(i,0,-1):
                V = V_stream[i-j]
                C = self.coefficients[j-1]
                V0 = np.zeros(len(PV[j-1]))
                
                for rf in range(len(V)):
                    V0[C[rf][0]] += V[rf] * C[rf][1]
                V_stream.append(V0)
            BV.append(V_stream[-1])
                
        #from the top
        BI = []
        for i in range(len(BV)-1, 0, -1):
            raw = R.backproject(BV[i], shape, fixation, normalize=False)
            normalized = np.uint8(np.true_divide(raw, self.norm_maps[i]))
            BI.append(normalized)
        BI.append(R.backproject(BV[0], shape, fixation)) #retinal
                
        return BI
        
""" Gaussian receptive field calculation """
def d_gauss(sigma, d, normalize=True):
    g = np.exp(-(d)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    if normalize: return g/np.sum(g)
    else: return g

"""
Params:
    L1 - a single, appropriately scaled tessellation (x,y) array. 
    L2 - a single, appropriately scaled tessellation that is a pyramid level
     higher than L1
    sigma_factor - (lambda) sigma scaling factor
    rf_fov - the field of view of each receptive field, defined as a factor of
     sigma
    
Returns:
    Gaussian pyramid coefficients between successive levels L2 and L1."""
def Gpyramid_coeffs(L0, L1, sigma_factor, rf_fov):
    L1_coeff = np.ndarray((len(L1)),dtype='object')
    
    #cdist
    DIST10 = utils.cdist_torch(L0, L1).numpy() #distances between the two levels
    DIST1 = utils.cdist_torch(L1, L1).numpy() #distances within L1
    
    dist_5 = np.mean(np.sort(DIST1)[:,1:6], axis=1)
    fov = rf_fov * dist_5 #TODO: dist5 * lambda for sumitha, * kratio (2.4) for u
    
    #L1_r = norm(L1, axis=1)
    L1_sigma = sigma_factor*dist_5
    
    for i in range(len(L1)):
        rf = np.where(DIST10[i] <= fov[i])[0]
        
        coeffs = (rf, d_gauss(L1_sigma[i], DIST10[i,rf]))
        L1_coeff[i] = coeffs
        
    return L1_coeff

"""
Params:
    L - a list or an array of pyramid tessellations, in order L0...Ln
    sigma_factor - (lambda) Gaussian rf pyramid scaling factor
    rf_fov - the field of view of each receptive field, defined as a factor of
     sigma
    
Returns:
    Pyramid class, eventually. For now a dict.
    Pyramid - {Tessellations:[L0...Ln], Coefficients:[L1... Ln]}
        L1 Coefficients: [N x tuple([L0 indices], [coeffs]) ]
            where N = len(L1 tessellation)
    
NOTES:
    Include base retinal parameters?"""
def Gpyramid_build(L, sigma_factor, rf_fov):
    P = {}
    P['Tessellations'] = L
    P['Coefficients'] = []
    
    for level in range(1,len(L)):
        c = Gpyramid_coeffs(L[level-1], L[level], sigma_factor, rf_fov)
        P['Coefficients'].append(c)
    
    return P

"""Pyramid computation function
NOTE: nans signify a node that is outside of the image frame. Nans need to be
omitted or they will propagate into the centre of the image via the pyramid"""
def Gpyramid(P, V0):
    PV = np.ndarray((len(P['Tessellations'])), dtype='object')
    PV[0] = V0
    
    for level in range(1,len(P['Tessellations'])):
        C = P['Coefficients'][level-1]
        nans = np.where(np.isnan(PV[level-1]))
        V = np.zeros(len(C))
        for node in range(len(C)):
            not_nan = np.where(np.logical_not(np.isin(C[node][0],nans)))
            coeff_i = C[node][0][not_nan]
            coeffs = C[node][1][not_nan]
            
            V[node] = np.sum(PV[level-1][coeff_i] * coeffs)
        PV[level] = V
    
    return PV

"""Produce normalization maps for each level of the pyramid""" 
def Gpyramid_norm(P, R):
    PV_norm = []
    for i in [1,2,3]:
        #Project unmodulated coefficients down a level (unit imagevector)
        V_stream = [np.ones(len(P['Tessellations'][i]))]
        
        #Down-propagate the projection to the retina
        for j in range(i,0,-1):
            Av = np.zeros(len(P['Tessellations'][j-1]))
            C = P['Coefficients'][j-1]
            for rf in range(len(P['Tessellations'][j])):
                Av[C[rf][0]] += C[rf][1] * V_stream[-1][rf]
            V_stream.append(Av)

        #Back-project an image
        GI = np.zeros((R.width, R.width))
        r = R.width/2.0
        V = V_stream[-1]
        for i in range(R.N - 1, -1, -1): 
            GI = utils.project(V[i] * R.coeff[0,i], GI, R.loc[i,:2][::-1] + r)
        
        norm = np.zeros(R._imsize[:2])
        norm = utils.project(GI, norm, R._normFixation)
        PV_norm.insert(0, norm)
        
    return PV_norm

"""A visualisation function for the pyramid
Params:
    P - A Gaussian pyramid
    PV - a pyramid vector
    R - an appropriate retina object

TODO: Requires using the same retina that was used to sample original V"""
def Gpyramid_backproject(P, PV, R, N, n=True):
    BV = []
    BV.append(PV[0])
    for i in [1,2,3]: 
        V_stream = [PV[i]]
        for j in range(i,0,-1):
            V = V_stream[i-j]
            C = P['Coefficients'][j-1]
            V0 = np.zeros(len(PV[j-1]))
            
            for rf in range(len(V)):
                V0[C[rf][0]] += V[rf] * C[rf][1]
            V_stream.append(V0)
        BV.append(V_stream[-1])
            
    #from the top
    BI = []
    for v in BV[::-1]:
        BI.append(R.backproject(v, R._imsize, R._fixation, normalize=n))
    return BI

#

#Load tessellations levels
pyr_path = join(datadir,"retinas","pyramid-4")
levels = ['50k_truetess.pkl', 'tess12k5.pkl', 'tess3125.pkl', 'tess781.pkl']

L0 = utils.loadPickle(join(pyr_path, levels[0]))
L1 = utils.loadPickle(join(pyr_path, levels[1]))
L2 = utils.loadPickle(join(pyr_path, levels[2]))
L3 = utils.loadPickle(join(pyr_path, levels[3]))

#Load retina, take a pic and sample
R = Retina(gpu=False)
R.loadLoc(join(datadir, "retinas", "ret50k_loc.pkl"))
R.loadCoeff(join(datadir, "retinas", "ret50k_coeff.pkl"))

#impath = "D:\\RETINA\\images\\Harmony_of_Dragons.jpg"
#impath = "D:\\RETINA\\images\\TEST.png"
impath = "D:\\RETINA\\images\\staircase.jpg"
img = np.float64(cv2.imread(impath, 0))
x = img.shape[1]/2
y = img.shape[0]/2
fixation = (y,x)

R.prepare(img.shape, fixation)
V = R.sample(img, fixation)
backproj = np.true_divide(R.backproject_last(n=False),R._gaussNorm)
utils.picshow(np.uint8(backproj), size=(15,15))

#Multiply up coarser tessellations Dmin to make them same scale as tess50k
d5 = 0.0021888014007064422 #fov_dist_5 from the raw 50k node tess
mean_rf = 1 #target fov_dist_5

L1 *= mean_rf/d5
L2 *= mean_rf/d5
L3 *= mean_rf/d5

L = [L0, L1, L2, L3]

#
#Sumitha's approach is to use lambda as k_width
lambda1 = 1.7321 #sumitha's lambda, w/ retina layer = 1
lambda2 = 1.6 * lambda1 #wider rfs
lambdaB = 0.5 * lambda1 #possibly 'your' lambda, w/ retinal layer = 0.5
rffov = 2.4 #k_ratio

#P = Gpyramid_build(L, lambda1, rffov) #narrow
P = {}
L = utils.loadPickle(join(pyr_path,"50K_pyr_narrow_tessellations.pkl"))
P["Tessellations"] = L
P['Coefficients'] = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_coeffs.pkl"))
N = Gpyramid_norm(P, R)

PV = Gpyramid(P, V)

P2 = Gpyramid_build(L, lambda2, rffov)
N2 = Gpyramid_norm(P2, R)

PV2 = Gpyramid(P2, V)

vistrue = Gpyramid_backproject(P, PV, R, N, n=False)
visnorm = Gpyramid_backproject(P, PV, R, N)

vistrue2 = Gpyramid_backproject(P2, PV2, R, N2, n=False)
visnorm2 = Gpyramid_backproject(P2, PV2, R, N2)

""" FRONTIERS: 3 diagrams (chop up)"""

#utils.writePickle(join(pyr_path, "50K_pyr_narrow_coeffs.pkl"), P["Coefficients"])
#utils.writePickle(join(pyr_path, "50K_pyr_narrow_normmaps.pkl"), N)
#utils.writePickle(join(pyr_path,"50K_pyr_narrow_tessellations.pkl"), L)

"""What do i do: 
    generate norm maps from a LoG pyramid, or subtract G_norm maps
    subtract imagevectors or just subtract un/normalized backprojections?
    TBH backprojection is expensive so subtracting PVs will be best
    and maybe generating the norm map, but that's for later
 """
 
#Converting to numpy obj array (actually turned out useless)
def objectify(coeffs):
    new_coeff = np.ndarray(len(coeffs), dtype='object')
    new_coeff[:] = coeffs
    for i in range(len(new_coeff)):
        for j in range(len(new_coeff[i])):
            tup = new_coeff[i][j]
            new_coeff[i][j] = np.ndarray(2, dtype='object')
            new_coeff[i][j][:] = [tup[0],tup[1]]
            
    return new_coeff

P['Coefficients'] = objectify(P['Coefficients'])

#LoG
#Produce LoG coeffs. Due to, uh, my data structure, it needs a loop
LoG_coeffs = deepcopy(P["Coefficients"]) #copy to keep indices
for L in range(len(P["Coefficients"])):
    for node in range(len(P["Coefficients"][L])):
        narrow = P["Coefficients"][L][node][1]
        wide = P2["Coefficients"][L][node][1]
        Cnew = (wide - narrow)
        Cnew /= np.sum(Cnew[np.where(Cnew > 0)]) #normalize???
        LoG_coeffs[L][node][1] = Cnew
        
LoG_P = {}
LoG_P['Tessellations'] = P['Tessellations']
LoG_P['Coefficients'] = LoG_coeffs

###

def posneg(im):
    neg = np.zeros_like(im)
    neg[np.where(im<0)] = 1
    utils.picshow(neg, size=(15,15))

LoG_N = Gpyramid_norm(LoG_P, R)
for i in range(len(LoG_N)): 
    utils.picshow(N[i])


LoG_PV = PV2 - PV #vis_true
LoG_PV2 = Gpyramid(LoG_P, V) #vis_true2
LoG_vis_true = Gpyramid_backproject(P, LoG_PV, R, N, n=False)
LoG_vis_true2 = Gpyramid_backproject(P, LoG_PV2, R, N, n=False)
#TODO1: the two ways of obtaining log_PV do not yield the same results.

"""
So the approach from LoG_PV is correct, and the backprojection should be
casted through and normalized by the narrow pyramid to generate the proper
visualisation (splatting after all) - as doing so through a DoG would re-construct
the laplace pattern instead of showing where the activations actually happen.

A good test for the pyramid is the spatial frequency human vision test. 
File test2.jpg in images - construct an image like that for your test, sample with 
pyramid/retina for good evaluations and include in paper.
"""

#visualize LoG pyramid
print("LoG pyramid (below)")
for i in [0,1,2]:
    A = -LoG_vis_true2[i]
    B = N[i]
    im = np.true_divide(A, B) +128
    utils.picshow(im, size=(15,15))
    print(A.max(), A.min(), len(np.unique(A)))
print("LoG pyramid (above)")



#visualize narrow pyramid
print("Narrow pyramid (below)")
for i in [0,1,2]:
    im = np.uint8(np.true_divide(vistrue[i],N[i]))
    utils.picshow(im, size=(15,15))
    print(len(np.unique(im)))
utils.picshow(visnorm[-1], size=(15,15))
print(len(np.unique(visnorm[-1])))
print("Narrow pyramid (above)")

#visualize wide pyramid
print("Wide pyramid (below)")
for i in [0,1,2]:
    im = np.uint8(np.true_divide(vistrue2[i], N2[i]))
    utils.picshow(im, size=(15,15))
    print(len(np.unique(im)))
utils.picshow(visnorm2[-1], size=(15,15))
print(len(np.unique(visnorm2[-1])))
print("Wide pyramid (above)")

"""NEXT:
- Figures!!!!!!!!!!
- Email paul showing the unnormalized images, ask him if this could be why the 
DC leakage is there according to the model u talked about or what (PV subtraction
causes quantization issues, no sane way to normalize it). 
- Wrap up into neat code, object-like.
- Does it work on the GPU?
- Apply sumitha's parameters (lambda 1:L0, 1.7...:L123). 
- Investigate the 'cortical color constancy model' params (lambda = 4)
""" 

#GAUSSIAN PYRAMIDS - make 2 with different rf sizes (factor 1.6 or 4)
#Define cortical gaussian filters. Pages 89,90
#Equation 3-30 (dist_c scaled by gamma=1.7321 in Gauss ret pyramid)
        
#'applied constant blurring in each layer = 1.7321 * initial blurring
#which gives 1.7321 * graph edge, or mean_dist_5. That's used to compute diameter
#of cortical support as well as gaussian sigma. In the retinal layer he maintains
#the value at 1, whereas your retina seems best with the value at 0.5 
#(lambda, or sigma_base). If his lambda fails, try 0.5 * 1.7321.