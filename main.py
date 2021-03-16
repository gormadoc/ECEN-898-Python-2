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
import cv2
import copy
import random
from timeit import default_timer as timer
from utils import *

def main(argv):
    usage = "main.py -i <image)> -s <scale> -n <noise> -m <minima>  -v <verbosity>"
    try:
        opts, args = getopt.getopt(argv, "hi:s:n:m:v:")
    except getopt.GetoptError:
        print(usage)
        print("Use\n\tmain.py -h\nto learn how to use this and run default settings")
        sys.exit(2)
    
    # assume a square kernel
    kernel_size = 9
    sigma = 3
    image = 'img/test.png'
    verbose = False
    connect = 4
    
    for opt, arg in opts:
        print(opt, arg)
        if opt == '-h':
            print(usage)
            print("Example usage: main.py -i coins -s x -n x -m x -v False")
            print("\tImage (str): elk, coins, coins_g, moon")
            print("\tScale (integer): ")
            print("\tNoise (integer): ")
            print("\tMinima (integer): ")
            print("\tVerbosity (str): True, False")
            print("No required arguments")
        elif opt == '-s':
            try:
                kernel_size = int(arg)
            except ValueError:
                print("Kernel size must be an integer")
                sys.exit(2)
        elif opt == '-t':
            try:
                threshold[0] = int(arg.rsplit(",")[0])
                threshold[1] = int(arg.rsplit(",")[1])
            except ValueError:
                print("Threshold values must be integers")
                sys.exit(2)
            except IndexError:
                print("Single-value threshold: {}".format(threshold[0]))
                threshold[1] = threshold[0]
        elif opt == '-i':
            if "elk" in arg.lower():
                image = 'img/elk.jpg'
            elif "coins_g" in arg.lower():
                image = 'img/coins2.png'
            elif "coins" in arg.lower():
                image = 'img/coins.png'
            elif "stone" in arg.lower():
                image = 'img/moon.jpg'
            else:
                print("Use\n\tmain.py -h\nto learn how to use the image argument; defaulting to elk")
        elif opt == '-m':
            try:
                minima = int(arg)
            except ValueError:
                print("Number of minima must be an integer")
                sys.exit(2)
        elif opt == '-v':
            if "false" in arg.lower() or "f" == arg.lower():
                verbose = False
            elif "true" in arg.lower() or "t" == arg.lower():
                verbose = True
            else:
                print("Use\n\tmain.py -h\nto learn how to use the verbosity argument; defaulting to False")
                verbose = False
    
    if kernel_size % 2 == 1:
        k = (kernel_size - 1)/2
    else:
        k = kernel_size / 2
    
    ''' Blur image '''
    # create Gaussian
    H = Gaussian2D(kernel_size, sigma)
    
    # load image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    # pad image
    start = timer()
    pad_amount = int(img.shape[0]/8)
    pad = pad_array(img, pad_amount)
    
    # blur image
    x = image_filter2d(pad, H)[pad_amount:pad_amount+img.shape[0], pad_amount:pad_amount+img.shape[1]]
    end = timer()
    print("\nTime taken for preprocessing: {0:.3f}".format(end - start))
        
    # find minima
    L = np.zeros(x.shape)+1
    change_in_label = True
    count = 0
    start = timer()
    while change_in_label is True:
        count = count + 1
        change_in_label = False
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] != 0:
                    for q in neighbors((i,j), x):
                        if x[i,j] > x[q]: 
                            # if p>q it cannot be minima
                            L[i,j] = 0
                            change_in_label = True
                        if x[i,j] == x[q] and L[q] == 0: 
                            # if p==q but q is not minima p cannot be either
                            L[i,j] = 0
                            change_in_label = True
    end = timer()
    print("\nTime taken to find minima: {0:.3f}".format(end - start))
    
    # save image with labeled minima
    if verbose:
        print("Minima-finding iterations: {}".format(count))
        xtc = cv2.cvtColor(np.array(255-copy.deepcopy(np.round(x, decimals=0))*255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] == 1:
                    xtc[i, j] = (0, 0, 255)
        cv2.imwrite('out/minima.png', xtc)
        
    # grow drain interiors
    start = timer()
    num_drains = grow_regions(L, unlabeled=1, connectedness=connect)
    end = timer()
    print("\nTime taken to grow drains: {0:.3f}".format(end - start))
    
    rand_bgr = np.random.randint(255, size=(num_drains+1, 3))
    if verbose:
        print("Number of drains: {}".format(num_drains))
        xtc = cv2.cvtColor(np.array(255-copy.deepcopy(np.round(x, decimals=0))*255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] > 0:
                    xtc[i, j] = (rand_bgr[int(L[i,j]), 0], rand_bgr[int(L[i,j]), 1], rand_bgr[int(L[i,j]), 2])
        cv2.imwrite('out/drains.png', xtc)
    
    # prep V structure
    Vl = []
    Vv = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # make lists of coordinates and their values
            if L[i,j] == 0:
                Vl.append((i,j))
                Vv.append(x[i,j])
                
    # sort the pixel coordinate list by the brightness
    V = [e for _,e in sorted(zip(Vv,Vl))]
    
    # fill basins
    start = timer()
    while V:
        p = V.pop(0)
        
        # find drain label
        for q in neighbors(p,L,connectedness=connect):
            try:
                if L[q] > 0:
                    L[p] = L[q]
                    break
            except IndexError:
                print("p: {}\tq: {}".format(p,q))
            
        
        # assign upstream pixels
        for q in neighbors(p,x,connectedness=connect):
            if q not in V:
                break
            slope = x[q] - x[p]
            # check if p downstream from q
            maximal = True
            for r in neighbors(q,x,connectedness=connect):
                if r == p:
                    continue
                if x[q] - x[r] > slope:
                    maximal = False
                    break
            if maximal:
                #print(slope)
                if L[q] > 0:
                    # already labeled
                    L[q] = num_drains + 1
                else:
                    L[q] = L[p]
        #print(len(V))
    end = timer()
    print("\nTime taken to grow basins: {0:.3f}".format(end - start))
        
    if verbose:
        xtc = cv2.cvtColor(np.array(copy.deepcopy(np.round(L, decimals=0)), dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] < num_drains+1:
                    xtc[i, j] = (rand_bgr[int(L[i,j])-1, 0], rand_bgr[int(L[i,j])-1, 1], rand_bgr[int(L[i,j])-1, 2])
                else:
                    xtc[i,j] = (0,0,0)
        cv2.imwrite('out/basins.png', xtc)
    
    
    
            
if __name__ == "__main__":
    main(sys.argv[1:])