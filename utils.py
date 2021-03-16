#!/usr/bin/env python

# Canny edge detection
'''
1. Gaussian filter smoothing (done)
2. Find intensity gradient
3. Pre-massage with gradient magnitude thresholding or lower bound cut-off suppression
4. Apply double threshold to find potential edges
5. Track edge by hysteresis: finalize detection by suppressing weak edges not connected to strong edges
'''

import sys, getopt, os
import math
import numpy as np


def pad_array(img, amount, method='replication'):
    method = method
    amount = amount
    t_img = np.array(img)
    re_img = np.zeros([img.shape[0]+2*amount, img.shape[1]+2*amount])
    re_img[amount:img.shape[0]+amount, amount:img.shape[1]+amount] = t_img
    if method == 'zero':
        pass # already that way
    elif method == 'replication':
        re_img[0:amount,amount:img.shape[1]+amount] = np.flip(img[0:amount, :], axis=0) # left
        re_img[-1*amount:-1, amount:img.shape[1]+amount] = np.flip(img[-1*amount:-1, :], axis=0) # right
        re_img[:, 0:amount] = np.flip(re_img[:, amount:2*amount], axis=1) # top
        re_img[:, -1*amount:] = np.flip(re_img[:, -2*amount:-amount], axis=1) # bottom
        
    return re_img
        
def image_filter2d(img, kernel):
    # establish useful values
    imx = img.shape[0]
    imy = img.shape[1]
    kx = kernel.shape[0]
    ky = kernel.shape[1]
    if kx % 2 == 1:
        center = [math.ceil(kx/2), math.ceil(ky/2)]
    else:
        center = [int(kx/2) + 1, int(ky/2) + 1]
        
    # pad arrays and put image in center
    re_img = np.zeros([imx+2*kx, imy+2*ky])
    pad_img = np.zeros([imx+2*kx, imy+2*ky])+np.max(np.max(img))/2
    pad_img[kx:imx+kx, ky:imy+ky] = img
    
    # Perform sum of products
    for row in range(kx, imx+kx):
        for col in range(ky, imy+ky):
            for a in range(0, kx):
                for b in range(0, ky):
                    re_img[row, col] = re_img[row,col] + pad_img[row+a-center[0]+1, col+b-center[1]+1]*kernel[a,b]
    return re_img[kx:imx+kx, ky:imy+ky]

def Gaussian2D(size, sigma):
    # simplest case is where there is no Gaussian
    if size==1:
        return np.array([[0,0,0],[0,1,0],[0,0,0]])

    # parameters
    peak = 1/2/np.pi/sigma**2
    width = -2*sigma**2
    
    # Gaussian filter
    H = np.zeros([size, size])

    # populate the Gaussian
    if size % 2 == 1:
        k = (size - 1)/2
        for i in range(1, size+1):
            i_part = (i-(k+1))**2
            for j in range(1, size+1):
                H[i-1, j-1] = peak*math.exp((i_part + (j-(k+1))**2)/width)
    else:
        k = size / 2
        for i in range(1, size+1):
            i_part = (i-(k+0.5))**2
            for j in range(1, size+1):
                H[i-1, j-1] = peak*math.exp((i_part + (j-(k+0.5))**2)/width)

    # normalize the matrix
    H = H / np.sum(np.concatenate(H))
    return H


def neighbors(pixelpos, image, connectedness=8):
    X,Y = image.shape
    x = pixelpos[0]
    y = pixelpos[1]
    n = []
    #print(X,Y,x,y)
    if connectedness == 8:
        for i in [-1, 0, 1]:
            # check within x bounds
            if x+i > -1 and x+i < X:
                #print(x+i)
                for j in [-1, 0, 1]:
                    # check within y bounds
                    if y+j > -1 and y+j < Y:
                        #print(y+j)mi
                        # p is not a neighbor of p
                        if x+i != 0 and y+j != 0:
                            n.append((x+i,y+j))
    elif connectedness == 4:
        if x > 0:
            n.append((x-1, y))
        if x < X-1:
            n.append((x+1, y))
        if y > 0:
            n.append((x, y-1))
        if y < Y-1:
            n.append((x, y+1))
    return n
    
    
def grow_regions(image, unlabeled=0, connectedness=8):
    X,Y = image.shape
    next_label = unlabeled + 1
    change_flag = True
    while change_flag:
        change_flag = False
        for i in range(0,X):
            for j in range(0,Y):
                if image[i,j] == unlabeled:
                    # start labeling a new region
                    current_label = next_label
                    next_label = next_label + 1
                    change_flag = True
                    
                    # grow that region
                    stack = [(i,j)]
                    while stack:
                        p = stack.pop(-1)
                        image[p] = current_label
                        for q in neighbors(p, image, connectedness):
                            if image[q] == unlabeled:
                                image[q] = current_label
                                stack.append(q)
                    break
            if change_flag:
                break
                
    # "normalize" such that max(image) = number of regions
    for i in range (0,X):
        for j in range(0,Y):
            if image[i,j] > 0:
                image[i,j] = image[i,j] - unlabeled
    return int(np.amax(image))
            
    
