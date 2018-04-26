# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 11:33:45 2017

A Python version for Sumitha's receptive field generation code from the file
wilsonretina5fine_8192s.m

@author: Piotr Ozimek
"""
#import sys
#sys.path.append('C:\Users\walsie\RETINA\python')

import numpy as np
from scipy.spatial import distance

#Gauss(sigma,x,y) function, 1D
def gauss(sigma,x,y,mean=0):
    d = np.linalg.norm(np.array([x,y]))
    return np.exp(-(d-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#Kernel(width,loc,sigma,x,y)
def gausskernel(width,loc,sigma):
    w = float(width)
    #location is passed as np array [x,y]
    k = np.zeros((width, width))    
    
    shift = (w-1)/2.0

    #subpixel accurate coords of gaussian centre
    dx = loc[0] - int(loc[0])
    dy = loc[1] - int(loc[1])    

#    m = np.sqrt((np.round(loc[0],decimals=1) - int(loc[0]))**2 + (np.round(loc[1],decimals=1) - int(loc[1]))**2)
    
    for x in range(width):
        for y in range(width):
            k[y,x] = gauss(sigma,(x-shift)-dx,(y-shift)-dy)
    
    return k    

def tess_scale(tessellation, neighbourhood, min_rf):
    #compute node density metric dist_5
    d = distance.cdist(tessellation, tessellation)
    s = np.sort(d)
    dist_5 = np.mean(s[:,1:neighbourhood], 1)                   #r
    
    #compute dist_5 for most central 5 nodes
    fov_dist_5 = np.mean(dist_5[:5])                            #closest_rfs
    
    #set central_dist_5 to min_rf (impose min_rf parameter)
    scaled = tessellation*(1/fov_dist_5)*min_rf
    
    return scaled

#Returns offsets (iirc)
def xy_sumitha(x,y,k_width):
    k_width = int(k_width) #this will change                 #d
    
    #if odd size mask -> round coordinates
    if k_width%2 != 0:
        cx = round(x)
        cy = round(y)
        
    #else if even size mask -> 1 decimal point coordinates (always .5)
    else:
        cx = round(x) + np.sign(x-round(x))*0.5
        cy = round(y) + np.sign(y-round(y))*0.5
    
    return cx, cy


"""A python version for Sumithas receptive field generation function. Output
should be nearly identical to the matlab version """
def rf_sumitha(tessellation, min_rf, sigma_ratio, sigma):
    rf_loc = np.zeros((len(tessellation),7))
    rf_coeff = np.ndarray((1,len(tessellation)),dtype='object')
    
    neighbourhood = 6 #5 closest nodes
    
    #compute node density metric dist_5
    d = distance.cdist(tessellation, tessellation)
    s = np.sort(d)
    dist_5 = np.mean(s[:,1:neighbourhood], 1)                   #r
    
    #set central_dist_5 to min_rf (impose min_rf parameter)
    fov_dist_5 = np.mean(dist_5[:5])                            #closest_rfs
    
    
    scaled = tessellation*(1/fov_dist_5)*min_rf
    rf_loc[:,:2] = scaled
    
    rf_loc[:,6] = np.ceil(sigma*(1/sigma_ratio)*(1/fov_dist_5)*min_rf*dist_5)
    
    for i in range(len(tessellation)):
        k_width = int(rf_loc[i,6])
        cx, cy = xy_sumitha(rf_loc[i,0], rf_loc[i,1], k_width)        
        
        rx = rf_loc[i][0] - cx
        ry = rf_loc[i][1] - cy
        loc = np.array([rx, ry])        
        
        #set rf_loc[i,0] (x) rf_loc[i,1] (y) appropriately.
        rf_loc[i,0] = cx
        rf_loc[i,1] = cy
        
        #place proper gaussian in rf_coeff[i]
        rf_coeff[0,i] = gausskernel(k_width,loc, k_width*sigma_ratio)
        rf_coeff[0,i] /= np.sum(rf_coeff[0,i])
        #but why does sumitha divied the gaussian by its sum? Ah to have a 'valid'
        #clipped gaussian that sums to 1?
          
    return rf_loc, rf_coeff

""" Completed function (I hope)
New rf gen function. New features + striving to maintain backwards compatibility
Params:
@tessellation is the raw node locations [x,y] array (currently produced in matlab)
@kernel_ratio is the ratio of kernel to local node density (dist_5)
@sigma_base is the base sigma, or global sigma scaling factor
@sigma_power is the power term applied to sigma scaling with eccentricity
@mean_rf sets the mean distance between the 20 most central nodes
@min_kernel imposes a minimum kernel width for the receptive fields (def: 3)

Returns:
@rf_loc - a num_of_nodes x 7 array that describes each node as follows:
[x, y, d, angle (radians), dist_5, rf_sigma, rf_width]
@rf_coeff - an array of gaussian receptive field kernels (variable size)
"""
def rf_ozimek(tessellation, kernel_ratio, sigma_base, sigma_power, mean_rf, min_kernel=3):
    rf_loc = np.zeros((len(tessellation), 7))
    rf_coeff = np.ndarray((1, len(tessellation)),dtype='object')
    
    neighbourhood = 6 #5 closest nodes
    
    print "rf generation - might take a while..."    
    
    #compute node density metric dist_5
    ##Break up cdist matrix into smaller pieces so they fit in ram
    length = len(tessellation)
    chunk = 5000
    num = length/chunk
    if length%chunk != 0:
        num += 1
    
    dist_5 = np.zeros(length, dtype='float64')
    print str(chunk) + " nodes in one chunk."
    for i in range(num):
        print "Processing chunk " + str(i)
        d = distance.cdist(tessellation[i*chunk:(i+1)*chunk], tessellation)
        s = np.sort(d)
        dist_5[i*chunk:(i+1)*chunk] = np.mean(s[:,1:neighbourhood], 1)
    
    #compute min dist_5 for most central 20 nodes
    fov_dist_5 = np.min(dist_5[:20])
    
    #set fov_dist_5 to mean_rf (impose mean_rf parameter)
    rf_loc[:,:2] = tessellation*(1/fov_dist_5)*mean_rf
    
    #Adjust dist_5 to reflect new scale
    dist_5 = dist_5*(1/fov_dist_5)*mean_rf
    
    #Insert angle to [3]
    rf_loc[:,3] = np.arctan2(rf_loc[:,1],rf_loc[:,0])
    
    #Insert dist_5 to [4]
    rf_loc[:,4] = dist_5

    print "All chunks done."
    
    ##determine sigmas
    #sigma_power decreases sigma if dist_5 < 1.0, so correct it
#    correction = 1-mean_rf if mean_rf < 1 else 0
    correction = 0
    rf_loc[:,5] = sigma_base * (dist_5+correction)**sigma_power
    
    
    #Use the same method as Sumitha for having even/odd kernels [compatibility]
    for i in range(len(tessellation)):
        #determine and insert kernel widths
        k_width = max(min_kernel, int(np.ceil(kernel_ratio*rf_loc[i,4])))
        rf_loc[i,6] = k_width
        
        cx, cy = xy_sumitha(rf_loc[i,0], rf_loc[i,1], k_width)
        
        #Obtain subpixel accurate offsets from kernel centre
        rx = rf_loc[i][0] - cx
        ry = rf_loc[i][1] - cy
        loc = np.array([rx, ry])
        
        #set [2] = eccentricity
        rf_loc[i,2] = np.linalg.norm(rf_loc[i,:2])
        
        #Set x, y, d
        rf_loc[i,0] = cx
        rf_loc[i,1] = cy
        
        #place proper gaussian in rf_coeff[i]
        rf_coeff[0,i] = gausskernel(k_width, loc, rf_loc[i,5])
        rf_coeff[0,i] /= np.sum(rf_coeff[0,i]) ###NORMALIZATION
    
    return rf_loc, rf_coeff


"""
function [Ind,M,closest_rfs]=wilsonretina5fine_8192s(retina,min_rf,sigma_ratio,s);
% scale for large RFs for gabors
% example sigma_ratio = 1/6

% min_rf_location=0.03; % on the 0.01th node of the sorted retina EDITED OUT 14.07.2005 
neighbourhood=6; % number of neighbours for density calculation [PO: set to 6]
%retina=retina*0.8;

for i=1:size(retina,1),
    D=distance(retina(i,:)',retina');
    d=sort(D);
    r(i)=mean(d(2:neighbourhood));  % average of the closest, because Paul said use density [PO: mean neighbourhood distance]
end

% EDITED OUT 14.07.2005
% r(1:round(size(r,2)*min_rf_location))=r(round(size(r,2)*min_rf_location));  % ignore smallest rfs in fovea. take minimum from rest
% closest_rfs=min(r(1:size(r,2))); 
closest_rfs=mean(r(1:5)); 

Ind(:,1:2)=retina*(1/closest_rfs)*min_rf; % scale retina so minimum rf gap is 1 pixel NEW " *min_rf "

Ind(:,7)=ceil(s*(1/sigma_ratio)*(1/closest_rfs)*min_rf*r'); % twice average rf gap in neighbourhood - scaled by ()
% (1/closest_rfs) is there because the rf sizes need to be scaled so min_rf

% EDITED OUT 14.07.2005
% Ind(1:min(find((Ind(:,7)==6)))-1,7)=5; % minimum rf size is 5

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Masks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[S,ignore]=size(Ind);

for i=1:S,
    
    % if i==1000, keyboard; end
    % size of mask
    d=Ind(i,7);  %MARKED LINE POZ
    
    if d/2-fix(d/2)>0, %odd size mask
        cx=round(Ind(i,1)); cy=round(Ind(i,2));
%         Ind(i,4)=cx-(d/2)+0.5; 
%         Ind(i,5)=cy-(d/2)+0.5; 
        X=Ind(i,1)-cx; Y=Ind(i,2)-cy;
        
    else % even size mask
        cx=round(Ind(i,1))+sign(Ind(i,1)-round(Ind(i,1)))*0.5;
        cy=round(Ind(i,2))+sign(Ind(i,2)-round(Ind(i,2)))*0.5;    
        %Ind(i,4)=cx-(d/2)+0.5;
        %Ind(i,5)=cy-(d/2)+0.5;
        X=Ind(i,1)-cx; Y=Ind(i,2)-cy;
    end
    
    Ind(i,1)=cx; Ind(i,2)=cy;
    % M{i}=mask(d,d/(6*1.6),-X,-Y,'dog',d/6);
    
    M{i}=mask(d,d*sigma_ratio,-X,-Y,'gau',d*sigma_ratio);
    M{i} = M{i}/sum(sum(M{i}));
    
end
"""
###################
"""
function M=mask(siz,sigma,X,Y,type,sigma2);
% siz can be even or odd
% X,Y denotes the subpixel distance of the centre of the
% gaussian from the centre of the mask
% M=mask(siz,sigma,X,Y,type,sigma2);

M=zeros(siz);

[x,y] = meshgrid(-(siz-1)/2:(siz-1)/2,-(siz-1)/2:(siz-1)/2);
x=x+X;
y=y+Y;

if type=='gau',
    h = (1/(2*pi*sigma2^2))*exp(-(x.*x + y.*y)/(2*sigma2*sigma2));
"""