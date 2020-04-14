#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 05:10:05 2020

@author: zhouhaoshuai
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from PIL import Image

def rgb_to_hsv(I):
    height, width, channel = I.shape
    res = np.zeros(I.shape)
    
    for i in range(height):
        for j in range(width):
            r, g, b = I[i][j]
            r, g, b = r/255., g/255., b/255.
            cmax = max([r, g, b])
            cmin = min([r, g, b])
            diff = cmax-cmin
            
            if cmax == cmin:
                h = 0
            elif cmax == r:
                h = (60*((g-b)/diff)+360)%360
            elif cmax == g:
                h = (60*((b-r)/diff)+120)%360
            elif cmax == b:
                h = (60*((r-g)/diff)+240)%360
            
            
            if not cmax:
                s = 0
            else:
                s = (diff/cmax)*100
            
            v = cmax*100
            
            res[i, j, 0] = h/2
            res[i, j, 1] = s*255/100
            res[i, j, 2] = v*255/100
    
    return res

def red_only(I):
    img_red_only = np.zeros([I.shape[0], I.shape[1]])
    
    for i in range(img_red_only.shape[0]):
        for j in range(img_red_only.shape[1]):
            if 0 < I[i, j, 0] < 5 and 50 < I[i, j, 1] < 255 and\
                20 < I[i, j, 2] < 255 or 175 < I[i, j, 0] < 180 and\
                50 < I[i, j, 1] < 255 and 20 < I[i, j, 2] < 255:
                    
                img_red_only[i, j] = 255
            else:
                img_red_only[i, j] = 0
                
    
    return img_red_only

def sobel(I):
    kernel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    res_horizontal = np.zeros(I.shape)
    for i in range(1, res_horizontal.shape[0]-1):
        for j in range(1, res_horizontal.shape[1]-1):
            res_horizontal[i, j] = np.sum(np.multiply(kernel_horizontal, I[i-1: i+2, j-1: j+2]))
            
    res_vertical = np.zeros(I.shape)
    for i in range(1, res_vertical.shape[0]-1):
        for j in range(1, res_vertical.shape[1]-1):
            res_vertical[i, j] = np.sum(np.multiply(kernel_vertical, I[i-1: i+2, j-1: j+2]))       
    
    res = np.sqrt(np.square(res_horizontal)+np.square(res_vertical))
    res = res*255.0/np.max(res)
    
    return res

def houghcircle(I, threshold, minRadius, maxRadius):
    h, w = I.shape
    accum = np.zeros([h+2*maxRadius, w+2*maxRadius, maxRadius-minRadius+1])
    
    theta = np.arange(0, 360)*np.pi/180
    
    for i in range(maxRadius, h+maxRadius):
        for j in range(maxRadius, w+maxRadius):
            if I[i-maxRadius, j-maxRadius]:
                for r in range(maxRadius-minRadius+1):
                    for angle in theta:
                        x, y = int((r+minRadius)*np.cos(angle)), int((r+minRadius)*np.sin(angle))
                        accum[i+x, j+y, r] += 1
    
    temp = accum[maxRadius: maxRadius+h, maxRadius: maxRadius+w]
    
    res = []
    for i in range(h):
        for j in range(w):
            for r in range(maxRadius-minRadius+1):
                v = temp[i, j, r]
                if v >= threshold:
                    res.append([v, i, j, r+minRadius])
    
    return res
                
def non_maximum_suppression(candidates, IOU_threshold):
    if candidates is None:
        return None
    candidates.sort(reverse=True)
    bounding_boxes = []

    for candidate in candidates:
        x1, y1, x2, y2 = candidate[2], candidate[1], candidate[4], candidate[3]
        for bounding_box in bounding_boxes:
            x3, y3, x4, y4 = bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]
            #calculate the IOU here
            x5, y5, x6, y6 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
            if x5 <= x6 and y5 <= y6:
                intersection = (x6-x5)*(y6-y5)
            else:
                intersection = 0
            union = (x2-x1)*(y2-y1)+(x4-x3)*(y4-y3)-intersection
            
            IOU = intersection / union

            if IOU >= IOU_threshold:
                break
        else:
            bounding_boxes.append([y1, x1, y2, x2])

    return bounding_boxes
    


def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    #convert the input image from rgb to hsv
    I_hsv = rgb_to_hsv(I)
    #select red pixels
    I_red_only = red_only(I_hsv)
    #edge detection with sobel
    I_sobel = sobel(I_red_only)
    I_sobel[I_sobel >= 127.5] = 255
    I_sobel[I_sobel < 127.5] = 0
    
    #circle detection with hough circle
    threshold, minRadius, maxRadius = 300, 3, 10
    circles = houghcircle(I_sobel, threshold, minRadius, maxRadius)
    rects = []
    for circle in circles:
        v, x, y, r = circle[0], circle[2], circle[1], circle[3]
        rects.append([v, y-r, x-r, y+r, x+r])
    
    #do non maximum suppression
    IOU_threshold = 0.2
    bounding_boxes = non_maximum_suppression(rects, IOU_threshold)
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes



# set the path to the downloaded data: 
data_path = '../hw1_data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../hw1_data/hw01_preds_hough'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)
    
    #show circles on the input image
    fig, ax = plt.subplots(1)
    ax.imshow(I)
    
    for bounding_box in preds[file_names[i]]:
        x1, y1, x2, y2 = bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]
        r = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='yellow', facecolor='none')
        ax.add_patch(r)
    
    plt.show()


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)

