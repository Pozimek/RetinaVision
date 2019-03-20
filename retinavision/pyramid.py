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

"""
03.02.2019 refactor plan
- Pyramid class:
    - validation function
    - separate DoG class that simplifies creating DoGs
Extensions:
    - extrema detection 
    - ->corners

- names: change all LoG and laplace to DoG
- robustness to different pyramidss
- so, uh, was/is objectify() needed?
- parameter investigations 
    sumitha's: lambda 1:L0, 1.7...:L123 (?)
    'cortical color constancy model' params (lambda = 4)
- validate on GPU
"""

class Pyramid:  
    #TODO: convert Coefficients into numpy object arrays! (?)
    def __init__(self, tess=0, coeffs=0, N=0, R=0):
        self.tessellations = tess
        self.coefficients = coeffs
        self.norm_maps = N
        self.retina = R
        self.PV = 0
        
        #TODO: descriptive vars
        self.levels = len(tess) if type(tess) == list else 0
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
          and the retina normalization image are only stored inside the\n\
          retina variable.")
    
    """Core sampling/pyramidizing function"""
    def sample(self, V0):        
        PV = np.ndarray(self.levels, dtype='object')
        PV[0] = V0
        
        for level in range(1,self.levels):
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
    
    def backproject_last(self, n=False):
        out = self.backproject(self.PV, self.retina._imsize, self.retina._fixation, n)
        return out    
    
    def backproject(self, PV, shape, fixation, n=False, laplace=False):
        R = self.retina #TODO cleanup variable namespace
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
        for v in BV[::-1]:
            BI.append(R.backproject_tight(v, R._imsize, R._fixation, normalize=n))
        return BI
    
    def visualize(self, BI, title, log=False):
        print(title + " (below)")
        for i in range(len(BI[:-1])): #:-1 skips the retinal level
            if log: 
                utils.picshow(np.true_divide(BI[i], self.norm_maps[i]))
            else: 
                utils.picshow(np.uint8(np.true_divide(BI[i], self.norm_maps[i])))
        print(title + " (above)")


"""Functions kept for posterity"""
class PyramidBuilder:
    def __init__(self):
        self.a = 0
        
    def generateTessellations50k(self):       
        pyr_path=join(datadir,"pyramid")
        #Load tessellations levels
        levels = ['50k_truetess.pkl', 'tess12k5.pkl', 'tess3125.pkl', 'tess781.pkl']
        
        L0 = utils.loadPickle(join(pyr_path, levels[0]))
        L1 = utils.loadPickle(join(pyr_path, levels[1]))
        L2 = utils.loadPickle(join(pyr_path, levels[2]))
        L3 = utils.loadPickle(join(pyr_path, levels[3]))
        
        #Multiply up coarser tessellations to make them same scale as tess50k
        d5 = 0.0021888014007064422 #fov_dist_5 from the raw 50k node tess
        mean_rf = 1 #target fov_dist_5
        
        L1 *= mean_rf/d5
        L2 *= mean_rf/d5
        L3 *= mean_rf/d5
        L = [L0, L1, L2, L3]
        
        #Sumitha's approach is to use lambda as k_width
        #lambda1 = 1.7321 #sumitha's lambda, w/ retina layer = 1
        #lambda2 = 1.6 * lambda1 #wider rfs
        #lambdaB = 0.5 * lambda1 #possibly 'your' lambda, w/ retinal layer = 0.5
        #rffov = 2.4 #k_ratio
        
        #P = Gpyramid_build(L, lambda1, rffov) #narrow
        #N = Gpyramid_norm(P, R)
        #utils.writePickle(join(pyr_path, "50K_pyr_narrow_coeffs.pkl"), P["Coefficients"])
        #utils.writePickle(join(pyr_path, "50K_pyr_narrow_normmaps.pkl"), N)
        #utils.writePickle(join(pyr_path,"50K_pyr_narrow_tessellations.pkl"), L)
        
        #P2 = Gpyramid_build(L, lambda2, rffov) #wide
        #N2 = Gpyramid_norm(P2, R)
        #utils.writePickle(join(pyr_path, "50K_pyr_wide_coeffs.pkl"), P2["Coefficients"])
        ##utils.writePickle(join(pyr_path, "50K_pyr_wide_normmaps.pkl"), N2)
        #utils.writePickle(join(pyr_path,"50K_pyr_wide_tessellations.pkl"), /
        #P2["Tessellations"])
        
        return L

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
    def Gpyramid_coeffs(self, L0, L1, sigma_factor, rf_fov):
        L1_coeff = np.ndarray((len(L1)),dtype='object')
        
        #cdist
        DIST10 = utils.cdist_torch(L0, L1).numpy() #distances between two levels
        DIST1 = utils.cdist_torch(L1, L1).numpy() #distances within L1
        
        dist_5 = np.mean(np.sort(DIST1)[:,1:6], axis=1)
        fov = rf_fov * dist_5 #TODO: dist5 * lambda for sumitha, * kratio (2.4) for u
        
        #L1_r = norm(L1, axis=1)
        L1_sigma = sigma_factor*dist_5
        
        for i in range(len(L1)):
            rf = np.where(DIST10[i] <= fov[i])[0]
            
            coeffs = (rf, utils.d_gauss(L1_sigma[i], DIST10[i,rf]))
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
    def Gpyramid_build(self, L, sigma_factor, rf_fov):
        P = {}
        P['Tessellations'] = L
        P['Coefficients'] = []
        
        for level in range(1,len(L)):
            c = self.Gpyramid_coeffs(L[level-1], L[level], sigma_factor, rf_fov)
            P['Coefficients'].append(c)
        
        return P

    """Produce normalization maps for each level of the pyramid""" 
    def Gpyramid_norm(self, tess, coeffs, R):
        PV_norm = []
        for i in [1,2,3]:
            #Project unmodulated coefficients down a level (unit imagevector)
            V_stream = [np.ones(len(tess[i]))]
            
            #Down-propagate the projection to the retina
            for j in range(i,0,-1):
                Av = np.zeros(len(tess[j-1]))
                C = coeffs[j-1]
                for rf in range(len(tess[j])):
                    Av[C[rf][0]] += C[rf][1] * V_stream[-1][rf]
                V_stream.append(Av)
    
            #Back-project an image
            GI = np.zeros((R.width, R.width))
            r = R.width/2.0
            V = V_stream[-1]
            for i in range(R.N - 1, -1, -1): 
                GI = utils.project(V[i] * R.coeff[0,i], GI, R.loc[i,:2][::-1] + r)
            
            norm = np.zeros((R.width, R.width))
            norm = utils.project(GI, norm, (R.width//2, R.width//2))
            PV_norm.insert(0, norm)
            
        return PV_norm


##Load retina, take a pic and sample
#R = Retina(gpu=False)
#R.loadLoc(join(datadir, "retinas", "ret50k_loc.pkl"))
#R.loadCoeff(join(datadir, "retinas", "ret50k_coeff.pkl"))
#
##impath = "D:\\RETINA\\images\\Harmony_of_Dragons.jpg"
##impath = "D:\\RETINA\\images\\TEST.png"
#impath = "D:\\RETINA\\images\\original.png"
#img = np.float64(cv2.imread(impath, 0))
#x = img.shape[1]/2
#y = img.shape[0]/2
#fixation = (y,x)
#
#R.prepare(img.shape, fixation)
#V = R.sample(img, fixation)
#backproj = np.true_divide(R.backproject_last(n=False),R._gaussNorm)
#utils.picshow(np.uint8(backproj), size=(10,10))

##
#PB = PyramidBuilder()
#pyr_path = join(datadir,"pyramid")
#L = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_tessellations.pkl"))
#lambda1 = 1.7321 #sumitha's lambda, w/ retina layer = 1
#lambda2 = 1.6 * lambda1 #wider rfs
#rffov = 2.4 #k_ratio
#
#P = PB.Gpyramid_build(L, lambda1, rffov) #narrow
#N = PB.Gpyramid_norm(P, R)
#utils.writePickle(join(pyr_path, "50K_pyr_narrow_coeffs.pkl"), P["Coefficients"])
#utils.writePickle(join(pyr_path, "50K_pyr_narrow_normmaps.pkl"), N)
#utils.writePickle(join(pyr_path,"50K_pyr_narrow_tessellations.pkl"), L)
#
#P2 = PB.Gpyramid_build(L, lambda2, rffov) #wide
#N2 = PB.Gpyramid_norm(P2, R)
#utils.writePickle(join(pyr_path, "50K_pyr_wide_coeffs.pkl"), P2["Coefficients"])
##utils.writePickle(join(pyr_path, "50K_pyr_wide_normmaps.pkl"), N2)
#utils.writePickle(join(pyr_path,"50K_pyr_wide_tessellations.pkl"), P2["Tessellations"])


######
#        
#pyr_path = join(datadir,"pyramid")
#L = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_tessellations.pkl"))
#C = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_coeffs.pkl"))
#L2 = utils.loadPickle(join(pyr_path, "50K_pyr_wide_tessellations.pkl"))
#C2 = utils.loadPickle(join(pyr_path, "50K_pyr_wide_coeffs.pkl"))
#
#PB = PyramidBuilder()
#N = PB.Gpyramid_norm(L, C, R)
#N2 = PB.Gpyramid_norm(L2, C2, R)
#
##
#
#utils.writePickle(join(pyr_path, "50K_pyr_narrow_normmaps.pkl"), N)
#utils.writePickle(join(pyr_path, "50K_pyr_wide_normmaps.pkl"), N2)

######
#'applied constant blurring in each layer = 1.7321 * initial blurring
#which gives 1.7321 * graph edge, or mean_dist_5. That's used to compute diameter
#of cortical support as well as gaussian sigma. In the retinal layer he maintains
#the value at 1, whereas your retina seems best with the value at 0.5 
#(lambda, or sigma_base). If his lambda fails, try 0.5 * 1.7321.
"""
A good test for the pyramid is the spatial frequency human vision test. 
File test2.jpg in images - construct an image like that for your test, sample 
with pyramid/retina for good evaluations and include in paper.
"""

##Testing object model
#
##Files
#pyr_path = join(datadir,"pyramid")
#L = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_tessellations.pkl"))
#L2 = utils.loadPickle(join(pyr_path, "50K_pyr_wide_tessellations.pkl"))
#N = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_normmaps.pkl"))
#N2 = utils.loadPickle(join(pyr_path, "50K_pyr_wide_normmaps.pkl"))
#C = utils.loadPickle(join(pyr_path, "50K_pyr_narrow_coeffs.pkl"))
#C2 = utils.loadPickle(join(pyr_path, "50K_pyr_wide_coeffs.pkl"))
#
##init
#narrow = Pyramid(tess = L, coeffs = C, N=N, R=R)
#wide = Pyramid(tess = L2, coeffs = C2, N=N2, R=R)
#
##process
#narrow_PV = narrow.sample(V)
#wide_PV = wide.sample(V)
#laplace = wide_PV - narrow_PV
#
##backproject
#narrow_vis = narrow.backproject_last()
#wide_vis = wide.backproject_last()
#laplace_vis = narrow.backproject(laplace, R._imsize, fixation)
#
##visualize
#narrow.visualize(narrow_vis, "Narrow Gaussian Pyramid")
#wide.visualize(wide_vis, "Wide Gaussian Pyramid")
#narrow.visualize(laplace_vis, "Laplacian Gaussian Pyramid", log=True)
#        
