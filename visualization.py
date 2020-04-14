import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

json_path = './preds.json'
image_path = '../hw1_data/RedLights2011_Medium'

with open(json_path) as f:
  preds = json.load(f)

img_names = sorted(os.listdir(image_path))
img_names = [f for f in img_names if '.jpg' in f]

for i in range(len(img_names)):
    I = Image.open(os.path.join(image_path, img_names[i]))
    I = np.asarray(I)

    fig, ax = plt.subplots(1)
    ax.imshow(I)

    for bounding_box in preds[img_names[i]]:
        x1, y1, x2, y2 = bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
    plt.show()