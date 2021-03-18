#!/usr/bin/env python

import sys, getopt, os
import math
import numpy as np
import cv2
import copy
import random
from timeit import default_timer as timer
from utils import *

def main(argv):
    usage = "main.py -i <image)> -s <sigma> -n <noise> -m <minima>  -v <verbosity> -k <kernel_size> -c <connectedness>"
    try:
        opts, args = getopt.getopt(argv, "hi:s:n:m:v:k:c:")
    except getopt.GetoptError:
        print(usage)
        print("Use\n\tmain.py -h\nto learn how to use this and run default settings")
        sys.exit(2)
    
    # assume a square kernel
    kernel_size = 9
    sigma = 3
    image = 'img/elk.png'
    verbose = False
    connect = 4
    minima = 1000
    mini_test = False
    
    for opt, arg in opts:
        print(opt, arg)
        if opt == '-h':
            print(usage)
            print("Example usage: main.py -i coins -s x -n x -m x -v False -c 4")
            print("\tImage (str): elk, coins, coins_g, moon")
            print("\tSigma (float): ")
            print("\tNoise (integer): ")
            print("\tMinima (integer): ")
            print("\tVerbosity (str): True, False")
            print("No required arguments")
        elif opt == '-s':
            try:
                sigma = float(arg)
                kernel_size = int(round(4 * sigma))
            except ValueError:
                print("Sigma must be a number")
                sys.exit(2)
        elif opt == '-i':
            if "elk" in arg.lower():
                image = 'img/elk.jpg'
            elif "coins_g" in arg.lower():
                image = 'img/coins2.png'
            elif "coins" in arg.lower():
                image = 'img/coins.png'
            elif "moon" in arg.lower():
                image = 'img/moon.jpg'
            elif "test" in arg.lower():
                image = 'img/test.png'
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
        elif opt == '-k':
            try:
                kernel_size = int(arg)
            except ValueError:
                print("Kernel size must be an integer")
                sys.exit(2)
        elif opt == '-c':
            try:
                if int(arg) == 4:
                    connect = 4
                elif int(arg) == 8:
                    connect = 8
                else:
                    connect = 4
            except:
                print("Connectedness must be '4' or '8'")
                
                
    file_suffix = '_m_' + str(minima) + '_c_' + str(connect) + '_s_' + str(sigma) + '_k_' + str(kernel_size) + '_' + image.rsplit("/", 1)[1].rsplit(".", 1)[0] + '.png'
            
    
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    
    ''' Blur image '''
    # create Gaussian
    H = Gaussian2D(kernel_size, sigma)
    
    # load image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    # pad image
    start = timer()
    pad_amount = 16
    pad = pad_array(img, pad_amount)
    
    # blur image
    x = (image_filter2d(pad, H)[pad_amount:pad_amount+img.shape[0], pad_amount:pad_amount+img.shape[1]]).round(decimals=0)
    end = timer()
    print("\nTime taken for preprocessing: {0:.3f}".format(end - start))
        
    if mini_test:
        x = np.array([[53,52,51,53,52,51,53,50,51],[49,50,49,51,40,41,39,41,40],[48,47,12,12,18,19,16,15,20],[46,41,12,12,19,20,17,15,16],[45,42,12,15,18,17,19,17,18],[46,44,43,44,41,16,18,20,19]])
        print(x)
        
    # find minima
    L = (np.zeros(x.shape)+1).astype(int)
    change_in_label = True
    count = 0
    start = timer()
    while change_in_label is True:
        count = count + 1
        change_in_label = False
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] != 0:
                    for q in neighbors(x, (i,j)):
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
    
    if mini_test:
        print(x)
        print(L)
    
    # save image with minima
    if verbose:
        print("Minima-finding iterations: {}".format(count))
        xtc = cv2.cvtColor(np.array(255-copy.deepcopy(np.round(x, decimals=0))*255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] == 1:
                    xtc[i, j] = (0, 0, 255)
        cv2.imwrite('out/minima' + file_suffix, xtc)
        
    # grow drain interiors
    start = timer()
    drains, drain_values = grow_regions(x, L, unlabeled=1, connectedness=connect)
    num_drains = len(drains)
    if num_drains > minima:
        drains = [e for _,e in sorted(zip(drain_values,drains))]
        drain_values = sorted(drain_values)
        while len(drains) > minima:
            drains.pop(-1)
            drain_values.pop(-1)
        for i in range(0, L.shape[0]):
            for j in range(0, L.shape[1]):
                if L[i,j] not in drains:
                    L[i,j] = 0
                else:
                    L[i,j] = drain_values.index(x[i,j])+1
        drains = []
        for i in range(1, len(drain_values)+1):
            drains.append(i)
        num_drains = len(drains)
    end = timer()
    print("\nTime taken to grow drains: {0:.3f}".format(end - start))
    
    if mini_test:
        print(x)
        print(L)
    
    rand_bgr = np.random.randint(255, size=(num_drains, 3))
    if verbose:
        print("Number of drains: {}".format(num_drains))
        xtc = cv2.cvtColor(np.array(255-copy.deepcopy(np.round(x, decimals=0))*255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] > 0:
                    xtc[i, j] = (rand_bgr[int(L[i,j])-1, 0], rand_bgr[int(L[i,j])-1, 1], rand_bgr[int(L[i,j])-1, 2])
        cv2.imwrite('out/drains' + file_suffix, xtc)
    
    # prep V structure
    start = timer()
    Vl = []
    Vv = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # make lists of coordinates and their values
            not_interior_flag = False
            for q in neighbors(L, (i,j), connectedness=connect):
                if L[q] == 0:
                    not_interior_flag = True
                    break
            if not_interior_flag:
                Vl.append((i,j))
                Vv.append(x[i,j])
                
    # sort the pixel coordinate list by the brightness
    V = [e for _,e in sorted(zip(Vv,Vl))]
    
    end = timer()
    print("\nTime taken to prep sorted queue: {0:.3f}".format(end - start))
    
    # fill basins
    start = timer()
    mask = copy.deepcopy(x)
    while V:
        p = V.pop(0)
        if L[p] == 0:
            continue
        
        for q in neighbors(mask, p, connect):
            if is_upstream(mask, q, p, connect) and q in V:
                if L[q] == 0:
                    L[q] = L[p]
                elif L[q] > 0 and L[q] != L[p]:
                    L[q] = num_drains+1
                    V.pop(V.index(q))
        # remove p from consideration without ruining function generality
        mask[p] = 255
        
    end = timer()
    print("\nTime taken to grow basins: {0:.3f}".format(end - start))
    
    if mini_test:
        print(x)
        print(L)
    
    xtc = cv2.cvtColor(np.array(copy.deepcopy(np.round(L, decimals=0)), dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            if L[i,j] in drains:
                xtc[i, j] = (rand_bgr[int(L[i,j])-1, 0], rand_bgr[int(L[i,j])-1, 1], rand_bgr[int(L[i,j])-1, 2])
            else:
                xtc[i,j] = (125,125,125)
    cv2.imwrite('out/basins' + file_suffix, xtc)
    
    
    
            
if __name__ == "__main__":
    main(sys.argv[1:])