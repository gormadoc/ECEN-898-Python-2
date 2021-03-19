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
    usage = "main.py -i <image)> -c <connectedness> -m <minima> -n <noise> -s <sigma> -k <kernel_size> -v <verbosity> "
    try:
        opts, args = getopt.getopt(argv, "hi:s:n:m:v:k:c:")
    except getopt.GetoptError:
        print(usage)
        print("Use\n\tmain.py -h\nto learn how to use this and run default settings")
        sys.exit(2)
    
    # assume a square kernel
    kernel_size = 20
    sigma = 5
    image = 'img/elk.jpg'
    verbose = False
    connect = 4
    minima = 1000
    mini_test = False
    noise = 0
    
    for opt, arg in opts:
        log("{0} {1}".format(opt, arg))
        if opt == '-h':
            print(usage)
            print("Example usage: main.py -i coins -s x -n x -m x -v False -c 4")
            print("\tImage (str): elk, coins, coins_g, moon")
            print("\tConnectedness (4 or 8): which pixels to consider neighbors")
            print("\tMinima (integer): number of drains to be kept, lowest minima preferred")
            print("\tNoise (float): percent of maximum pixel for range of noise added")
            print("\tSigma (float): blurring strength")
            print("\tKernel size(odd integer): size of blurring kernel, calculated from sigma if not specified")
            print("\tVerbosity (str): True, False")
            print("No required arguments")
            print("To avoid blurring, sigma can be set to zero while not specifying kernel size or kernel size can be set to zero")
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
            elif "book" in arg.lower():
                mini_test = True
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
        elif opt == '-n':
            try:
                noise = float(arg)
            except ValueError:
                print("Noise must be a number 0-100")
                sys.exit(2)
            if noise < 0 or noise > 100:
                print("Noise must be a number 0-100")
                sys.exit(2)
                
    # prepare file names
    file_suffix = image.rsplit("/", 1)[1].rsplit(".", 1)[0] + '_c_' + str(connect) + '_m_' + str(minima) + '_n_' + str(noise) + '_s_' + str(sigma) + '_k_' + str(kernel_size) 
    logfile = 'out/' + file_suffix + '_info.txt'
    open(logfile, 'w').close()
    
    # Force kernels to be odd
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    
    # load image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    # pre-process image
    start = timer()
    if kernel_size > 1 or noise != 0.0:
        # add noise
        if noise != 0.0:
            pc_noise = np.max(img)*noise/100
            noise_arr = np.random.randint(-1*pc_noise, high=pc_noise, size=img.shape)
            img = img + noise_arr
            
            for i in range(0,img.shape[0]):
                for j in range(0,img.shape[1]):
                    if img[i,j] < 0:
                        img[i,j] = 0
        
        # pad image, blur image, and crop to original image
        if kernel_size > 1:
            pad_amount = int((kernel_size-1))
            pad = pad_array(img, pad_amount)
            H = Gaussian2D(kernel_size, sigma)
            img = (image_filter2d(pad, H)[pad_amount:pad_amount+img.shape[0], pad_amount:pad_amount+img.shape[1]]).round(decimals=2)
    
    x = img
    end = timer()
    log("\nTime taken for preprocessing: {0:.3f}".format(end - start), file=logfile)
       
    # book example for testing
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
    log("\nTime taken to find minima: {0:.3f}".format(end - start), file=logfile)
    
    if mini_test:
        print(x)
        print(L)
    
    # save image with minima
    log("Minima-finding iterations: {}".format(count), file=logfile)
    if verbose:
        xtc = cv2.cvtColor(np.array(255-copy.deepcopy(np.round(x, decimals=0))*255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] == 1:
                    xtc[i, j] = (0, 0, 255)
        cv2.imwrite('out/' + file_suffix + '_minima.png', xtc)
        
    # grow drain interiors
    start = timer()
    drains = grow_regions(x, L, unlabeled=1, connectedness=connect)
    num_drains = len(drains)
    log("Number of drains found: {}".format(num_drains), file=logfile)
    
    # discard larger drains over requested number of minima
    if num_drains > minima:
        drains = {key: value for key, value in sorted(drains.items(), key=lambda item: item[1])}
        drain_keys = list(drains.keys())
        for key in drain_keys:
            if drain_keys.index(key)+1 > minima:
                drains.pop(key)
            
        for i in range(0, L.shape[0]):
            for j in range(0, L.shape[1]):
                if L[i,j] not in drains:
                    L[i,j] = 0

        num_drains = len(drains)
    drain_keys = list(drains.keys())

    end = timer()
    log("\nTime taken to grow drains: {0:.3f}".format(end - start), file=logfile)
    
    if mini_test:
        print(x)
        print(L)
    
    rand_bgr = np.random.randint(255, size=(num_drains, 3))
    log("Number of drains kept: {}".format(num_drains), file=logfile)
    if verbose:
        xtc = cv2.cvtColor(np.array(255-copy.deepcopy(np.round(x, decimals=0))*255, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                if L[i,j] > 0:
                    xtc[i, j] = (rand_bgr[drain_keys.index(L[i,j]), 0], rand_bgr[drain_keys.index(L[i,j]), 1], rand_bgr[drain_keys.index(L[i,j]), 2])
        cv2.imwrite('out/' + file_suffix + '_drains.png', xtc)
    
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
    log("\nTime taken to prep sorted queue: {0:.3f}".format(end - start), file=logfile)
    
    # fill basins
    start = timer()
    mask = copy.deepcopy(x)
    while V:
        p = V.pop(0)
        if L[p] == 0:
            continue
        
        for q in neighbors(mask, p, connect):
            if q in V and is_upstream(mask, q, p, connect):
                if L[q] == 0:
                    L[q] = L[p]
                elif L[q] > 0 and L[q] != L[p]:
                    L[q] = 10000
                    V.pop(V.index(q))
        # remove p from consideration without ruining function generality
        mask[p] = 255
        
    end = timer()
    log("\nTime taken to grow basins: {0:.3f}".format(end - start), file=logfile)
    
    if mini_test:
        print(x)
        print(L)
    
    xtc = cv2.cvtColor(np.array(copy.deepcopy(np.round(L, decimals=0)), dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            if L[i,j] in drain_keys:
                xtc[i, j] = (rand_bgr[drain_keys.index(L[i,j]), 0], rand_bgr[drain_keys.index(L[i,j]), 1], rand_bgr[drain_keys.index(L[i,j]), 2])
            elif L[i,j] == 10000:
                xtc[i,j] = (0,100,255)
            else:
                xtc[i,j] = (125,125,125)
    cv2.imwrite('out/' + file_suffix + '_basins.png', xtc)
    
    
    
            
if __name__ == "__main__":
    main(sys.argv[1:])