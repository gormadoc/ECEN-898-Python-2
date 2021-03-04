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
    kernel_size = 5
    sigma = 1/math.sqrt(3)
    threshold = [10, 15]
    image = 'img/test.png'
    minima = 3
    verbose = False
    operator = "sobel"
    
    for opt, arg in opts:
        print(opt, arg)
        if opt == '-h':
            print(usage)
            print("Example usage: main.py -i elk -s x -n x -m x -v False")
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
    pad_amount = int(img.shape[0]/4)
    pad = pad_array(img, pad_amount)
    if verbose:
        pass
        #cv2.imwrite('pad_' + str(pad_amount) + '_' + image.rsplit("/", 1)[1].rsplit(".", 1)[0] + '.png', pad)
    
    # blur image
    x = image_filter2d(pad, H)[pad_amount:pad_amount+img.shape[0], pad_amount:pad_amount+img.shape[1]]
    if verbose:
        pass
        #cv2.imwrite('blurred_' + str(kernel_size) + '_' + image.rsplit("/", 1)[1].rsplit(".", 1)[0] + '.png', x)
        
    # find our drains
    drains = [ [] for _ in range(minima)]
    xt = copy.deepcopy(np.round(x, decimals=0))
    for i in range(0, minima):
        # find indices and value of most minimum point
        mind = np.unravel_index(np.argmin(xt, axis=None), xt.shape)
        val = xt[mind[0], mind[1]]
        print(val)
        
        # set up lists to blob from
        inlist = [mind]
        outlist = []
        while len(inlist) != 0:
            px = inlist.pop()
            if px not in outlist and px not in inlist:
                print("({0}, {1}): {2}".format(px[0], px[1], xt[px[0], px[1]]))
                # we don't need to evaluate this point again
                outlist.append((px[0], px[1]))
                
                # check if the popped point should be in the drain
                if xt[px[0], px[1]] == val:
                    drains[i].append(px)
                    xt[px[0], px[1]] = 255
                    
                    # add neighboring points (8-connectedness)
                    for j in [px[0]-1, px[0], px[0]+1]:
                        if j < 0 or j > xt.shape[0]-1:
                            continue
                        for k in [px[1]-1, px[1], px[1]+1]:
                            if not(j == 0 and k == 0) and k > 0 and k < xt.shape[1]:
                                inlist.append((j,k))
    
    if verbose:
        cv2.imwrite('out/drains_test.png', xt)
        xtc = np.array(xt * 255, dtype = np.uint8)
        xtc = cv2.cvtColor(xtc, cv2.COLOR_GRAY2RGB)
        i = 0
        for d in drains:
            for p in d:
                xtc[p[0], p[1]] = (255 / minima * (i+1), 0, 255)
            i = i + 1
        cv2.imwrite('out/drains_identified.png', xtc)
    
    ''' Calculate intensity gradient '''
    grady = np.array([[0,0,0],[0,1,0],[0,0,0]]) # this shouldn't be used but is here just in case
    if "sobel" in operator:
        grady = 1/8*np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    elif "prewitt" in operator:
        grady = 1/6*np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    gradx = np.transpose(grady) # we're dealing with symmetric kernels
    
    dy = image_filter2d(x, grady)
    dx = image_filter2d(x, gradx)
    
    
    
            
if __name__ == "__main__":
    main(sys.argv[1:])