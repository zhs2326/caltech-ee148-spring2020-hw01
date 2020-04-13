import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from PIL import Image

def resize(I, factor):
    factor = max(factor, 1/I.shape[0], 1/I.shape[1])
    resized_image = np.zeros([int(factor*I.shape[0]), int(factor*I.shape[1]), I.shape[2]])

    #do interpolation horizontally
    for i in range(I.shape[0]):
        for j in range(I.shape[2]):
            xp = [int((k+1)*factor)-1 for k in range(I.shape[1])]
            fp = [I[i, k, j] for k in range(I.shape[1])]
            resized_image[int(factor*(i+1))-1, :, j] = np.interp([k for k in range(resized_image.shape[1])], xp, fp)

    #do interpolation vertically
    for j in range(resized_image.shape[1]):
        for k in range(I.shape[2]):
            xp = [int((i+1) * factor)-1 for i in range(I.shape[0])]
            fp = [resized_image[int(factor*(i+1))-1, j, k] for i in range(I.shape[0])]
            resized_image[:, j, k] = np.interp([i for i in range(resized_image.shape[0])], xp, fp)


    resized_image = resized_image.astype(int)
    return resized_image


def normalize(I):
    I = np.copy(I)
    mean = np.mean(I)
    std = np.std(I)

    if std:
        I = (I-mean)/std
    else:
        I = np.ones(I.shape)

    return I

def matched_filtering(window, template):
    assert window.shape == template.shape

    h, w, c = window.shape
    s = np.sum(np.multiply(window, template))/(h*w*c)

    return s

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
            if x2 >= x3 and y2 >= y3:
                intersection = (x2-x3)*(y2-y3)
                union = (x2-x1)*(y2-y1)+(x4-x3)*(y4-y3)-intersection
            elif x4 >= x1 and y2 >= y3:
                intersection = (x4 - x1) * (y2 - y3)
                union = (x2 - x1) * (y2 - y1) +(x4-x3)*(y4-y3) - intersection
            elif x2 >= x3 and y4 >= y1:
                intersection = (x2 - x3) * (y4 - y1)
                union = (x2 - x1) * (y2 - y1) +(x4-x3)*(y4-y3) - intersection
            elif x4 >= x1 and y4 >= y1:
                intersection = (x4 - x1) * (y4 - y1)
                union = (x2 - x1) * (y2 - y1) +(x4-x3)*(y4-y3) - intersection
            IOU = intersection / union

            if IOU >= IOU_threshold:
                break
        else:
            bounding_boxes.append([y1, x1, y2, x2])

    return bounding_boxes




def detect_red_light(I, template, ax):
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
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    '''
    box_height = 8
    box_width = 6
    
    num_boxes = np.random.randint(1,5) 
    
    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)
        
        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width
        
        bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    '''
    #different template size
    sizes = [0.25, 0.5, 1]

    for size in sizes:
        resized_template = resize(template, size)


        #fig2, ax2 = plt.subplots(1)
        #ax2.imshow(resized_template)
        #plt.show()

        #normalize template
        normalized_template = normalize(resized_template)

        img_h, img_w, img_c = I.shape
        template_h, template_w, template_c = normalized_template.shape

        matched_filtering_threshold = 0.8
        IOU_threshold = 0.5
        candidates = []

        for i in range(img_h-template_h+1):
            for j in range(img_w-template_w+1):
                #get the sliding window and normalize the window
                window = I[i: i+template_h, j: j+template_w]
                normalized_window = normalize(window)

                #do matched filtering here
                correlation =  matched_filtering(normalized_window, normalized_template)
                if correlation > matched_filtering_threshold:
                    candidates.append([correlation, i, j, i+template_h-1, j+template_w-1])

        #do non maximum suppression
        #bounding_boxes.extend(non_maximum_suppression(candidates, IOU_threshold))
        bounding_boxes.extend(candidates)

    # do non maximum suppression
    bounding_boxes = non_maximum_suppression(bounding_boxes, IOU_threshold)


    #draw bounding box
    for bounding_box in bounding_boxes:
        x1, y1, x2, y2 = bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

    print(bounding_boxes)

    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = '../hw1_data/RedLights2011_Medium'

#template image path
template_path = '../hw1_data/template/template.jpg'

# set a path for saving predictions: 
preds_path = '../hw1_data/hw01_preds'
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

    #read the template and convert to numpy array
    template = Image.open(template_path)
    template = np.asarray(template)

    fig, ax = plt.subplots(1)
    ax.imshow(I)




    preds[file_names[i]] = detect_red_light(I, template, ax)

    plt.show()

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
