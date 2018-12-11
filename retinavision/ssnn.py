#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:06:11 2018

Self Similar Neural Network, in PyTorch

NOTE: PyTorch for Windows runs only on Python 3. This script was written in
Python 3.6.
@author: Piotr Ozimek
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import torch as torch
from numpy.random import uniform as rand
from time import perf_counter
from IPython.display import display
import pickle
import winsound

def writePickle(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)

def memshow():
    a = torch.cuda.memory_allocated()
    c = torch.cuda.memory_cached()
    r = torch.cuda.max_memory_allocated()-torch.cuda.memory_allocated()
    print("Allocated:", a)
    print("Cached:", c)
    print("Remaining (alloc):", r, "(", r/torch.cuda.max_memory_allocated(),")")
    
def NN(x, y):
    """Nearest Neighbour in Pytorch
    Returns: array of indices of the nearest neighbour to each element of x
    Legacy version - use utils.cdist_torch() instead"""
    xs = x.shape[0]
    ys = y.shape[0]
    Y = torch.from_numpy(y).to("cuda")
    X = torch.from_numpy(x).to("cuda")
    
    Xd = torch.zeros(ys,xs,2,device="cuda")
    for i in range(ys):
        Xd[i] = X - Y[i]
        
    del X, Y
    
    tnorm = torch.norm(Xd,2,2)    #OOM here
    del Xd
    
    #uncomment for 2nd closest neighbour (when x=y)
#    am = torch.argmin(tnorm,dim=1)
#    tnorm[range(xs),am] = 3.40282e+38 #near the max value for float32
    closest = torch.argmin(tnorm,dim=1)
    out = closest.to("cpu")
    del tnorm, closest

    return out

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def tessplot(T, s = (10,10), c = None, cmap=None, size=1, axis='on'):    
    plt.figure(figsize = s)
    if axis=='off': plt.axis('off')
    plt.scatter(T[:,0], T[:,1], s = size, c = c, cmap=cmap)
    plt.show()

def shownodes(T, i, s = (8,8)):
    n = T[i]
    plt.figure(figsize = s)
    plt.scatter(T[:,0], T[:,1], s = 3)
    plt.scatter(n[:,0], n[:,1], s = 5, c = 'r')
    if len(n) == 1:
        plt.ylim(n[0,1]-50, n[0,1]+50)
        plt.xlim(n[0,0]-50, n[0,0]+50)
    plt.show()
    
def imshow(pic, size=(10,10)):    
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(pic, interpolation='none')
    plt.show()

def alarm():
    winsound.Beep(600,400)
    winsound.Beep(600,400)
    winsound.Beep(600,400)

def SSNN(N, name=None, fps=90, fov=0.1, max_iter=20000, init_alpha=0.1, end_alpha=0.0005, V = False):
    """Self Similar Neural Network algorithm, generating retinal tessellations
    N: # of nodes, fov: proportion of foveal region, v: save a video of the process"""
    c_iter = int(max_iter/4) #initial iterations with a constant alpha
    
    #Annealed learning rate
    ALPHA = np.zeros(max_iter)
    ALPHA[:c_iter] = init_alpha
    a_gradient = (end_alpha-init_alpha)/(max_iter - max_iter/4)
    
    for i in range(c_iter, max_iter):
        ALPHA[i] = ALPHA[i-1]+a_gradient
    
    r = rand(size = (N))
    th = 2 * np.pi * rand(size=(N))
    W = np.dstack(pol2cart(r, th))[0]
    
    if V:
        if name is None: writer = imageio.get_writer('D://RETINA//out//ssnn-{0}.mp4'.format(N), fps=fps)
        else: writer = imageio.get_writer('D://RETINA//out//{0}.mp4'.format(name), fps=fps)
    
    for i in range(max_iter):    
        a = ALPHA[i]
        print("Iteration {0}".format(i))
        r, th = cart2pol(W[:,0], W[:,1])
        
        #dilation, rotation and translation factors
        dil = np.exp((2 * rand()-1) * np.log(8))
        rot = rand() * 2 * np.pi
        delta_theta = rand() * 2 * np.pi
        delta_r = rand() * fov
        dx = np.cos(delta_theta) * delta_r
        dy = np.sin(delta_theta) * delta_r
        
        """Implementing transformations"""
        #XXX: the order in sumitha's matlab code is different from the paper (dil trans rot in code)
        #Dilation
        I = np.dstack(pol2cart(r*dil, th))[0] #TODO: modify this for a flat fovea
        
        #Translation
        I[:,0] = I[:,0] + dx
        I[:,1] = I[:,1] + dy    
        r, th = cart2pol(I[:,0], I[:,1])
        #Rotation
        th += rot
        
        #Cull outliers
        outside = np.where(r>1)
        r = np.delete(r,outside)
        th = np.delete(th,outside)
        I = np.dstack(pol2cart(r,th))[0]
        
        if I.shape[0] != 0:
            #Nearest neighbours
            D = NN(W, I)
            #Update W
            for i in range(N):
                closest = np.where(D==i)[0]
                if len(closest) != 0:
#                    print(closest, I[closest])
#                    z = (np.ones((len(closest),1)) * W[i]) - I[closest]
#                    print(z)
                    v = np.sum((np.ones((len(closest),1)) * W[i]) - I[closest], axis=0)
                    W[i] = W[i] - a * v
        
        if V:
            #Generate a scatter image
            plt.switch_backend('agg')    
            fig = plt.figure(figsize = (10,10))
            _ = plt.scatter(W[:,0], W[:,1], s = 1)
            _ = plt.axis('off')
        
            fig.canvas.draw()    
            width, height = fig.get_size_inches() * fig.get_dpi()    
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            del(fig)
            #Append to file
            writer.append_data(frame)
    
    if V: writer.close()
    torch.cuda.empty_cache()
    
    alarm()
    return W

"""
initial_alpha=0.1;  alpha=initial_alpha; final_alpha=0.0005;

Alpha(1:max_iterations/4)=initial_alpha;
alpha_gradient=(final_alpha-initial_alpha)/(max_iterations-max_iterations/4);

for iterations=(max_iterations/4)+1:max_iterations,
    Alpha(iterations)=Alpha(iterations-1)+alpha_gradient;

r=rand(n,1); 
th=2*pi*rand(n,1);
[W(:,1) W(:,2)]=pol2cart(th,r);
I=W;

for iterations=1:max_iterations,
    alpha=Alpha(iterations);
    
    %%%%%%%%%%%% rotations %%%%%%%%%%%%%%%
    [th,r]=cart2pol(W(:,1),W(:,2));
    
    %%%%%%%%%%%% dilations %%%%%%%%%%%%%%%
    d=exp((2*rand-1)*log(8));   
    clear I;
    
    %%%%%%%%%%%% translations %%%%%%%%%%%% 
    [I(:,1) I(:,2)]=pol2cart(th,r*d); 
    
    % translation in a radial direction
    delta_theta=2*rand*pi; delta_rho=rand*fovea;
    
    I(:,1)=I(:,1)+cos(delta_theta)*delta_rho;
    I(:,2)=I(:,2)+sin(delta_theta)*delta_rho;
    
    %%%%%%%%%%%% thresholding %%%%%%%%%%%%
    [th,r]=cart2pol(I(:,1),I(:,2));
    th=th+rand*2*pi; 
    
    %%%%%%%%%%%% cull %%%%%%%%%%%%%%%%%%%%%
    outside=find(r>1);
    r(outside)=[];
    th(outside)=[];
    clear I;
    [I(:,1) I(:,2)]=pol2cart(th,r);
    
    %%%%%%%%%%%% calculate distance matrix %%%%%%%%%%%%%%%%%%
    clear closest_weight;
    [s,ignore]=size(I);
    
    if s~=0
        for i = 1:s,
            [ignore,closest_weight(i)] = min(sqrt(sum(((ones(n,1)*I(i,:)-W)').^2)));
        
        %%%%%%%%%%%% update %%%%%%%%%%%%%%%%%%
        
        for i=1:n,  % find input vectors for which W(i,:) is the closest
            closest_input=find(closest_weight==i);
            
            if ~isempty(closest_input),
                [ignore,s]=size(closest_input);
                v=sum((ones(s,1)*W(i,:)-I(closest_input,:))); % overall distance vector
                W(i,:)=W(i,:)-alpha*v;
        
        [th,r]=cart2pol(W(:,1),W(:,2));
        outside=find(r>1);
        r(outside)=1;
        [W(:,1) W(:,2)]=pol2cart(th,r);
    
R=W;
"""