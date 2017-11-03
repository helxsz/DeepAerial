
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from skimage import img_as_float, io
from sklearn.metrics import confusion_matrix
import itertools
import argparse
import os
from tqdm import tqdm


def label_to_pixel(label):
    """ Converts the numeric label from the ISPRS dataset into its RGB encoding

    Args:
        label (int): the label value (numeric)

    Returns:
        numpy array: the RGB value
    """
    codes = [[255, 255, 255],
             [0, 0, 255],
             [0, 255, 255],
             [0, 255, 0],
             [255, 255, 0],
             [255, 0, 0],
             [0, 0 , 0]]
    return np.asarray(codes[int(label)])

 # In[6]:

# Simple sliding window function
def sliding_window(top, step=10, window_size=(20,20)):
    """Extract patches according to a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        list: list of patches with window_size dimensions
    """
    # slide a window across the image
    for x in xrange(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in xrange(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
 
def count_sliding_window(top, step=10, window_size=(20,20)):
    """Count the number of patches in a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        int: patches count in the sliding window
    """
    c = 0
    # slide a window across the image
    for x in xrange(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in xrange(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c
    
def grouper(n, iterable):
    """ Groups elements in a iterable by n elements
    Args:
        n (int): number of elements to regroup
        iterable (iter): an iterable
    Returns:
        tuple: next n elements from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def process_votes(prediction):
    """ Returns RGB encoded prediction map
    Args:
        votes (array): full prediction from the predict function
    Returns:
        array: RGB encoded prediction
    """
    rgb = np.zeros(prediction.shape[:2] + (3,), dtype='uint8')
    for x in xrange(prediction.shape[0]):
        for y in xrange(prediction.shape[1]):
            rgb[x,y] = np.asarray(label_to_pixel(np.argmax(prediction[x,y])))
    return rgb

def findcars(image, step=32, patch_size=(512,512)):

    BATCH_SIZE = 10
    votes = np.zeros(image.shape[:2] + (6,))
    for coords in tqdm(grouper(BATCH_SIZE, sliding_window(image, step, patch_size)), total=count_sliding_window(image, step, patch_size)/BATCH_SIZE + 1):
        image_patches = []

        for x,y,w,h in coords:
            image_patches.append(image[x:x+w, y:y+h])
            print(x,x+w, y,y+h)

        bounding_cars = parseImage(image_patches,boundaries[0],False)

    return votes



boundaries = [
    (  [30, 100, 100 ],[32, 255, 255 ]),  #  yelow 255,255,0
 
   # (  [ 71, 100, 100 ],[83, 255, 255 ]), #hsv    green
]

def parseImage(images, boundary , show):
	lower, upper= boundary[0], boundary[1]
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	#print lower, upper	 

	for image in images:

		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)	
		mask = cv2.inRange(hsv, lower, upper)
		output = cv2.bitwise_and(image, image, mask = mask)
		
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_NONE)
		for i in range(len(contours)):
			if len(contours[i]) > 20:
				x, y, w, h = cv2.boundingRect(contours[i])
				cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
				print 'found the car at position:', (x, y),(x+w, y+h)            
				#positions.append([x, y, x+w, y+h])
		#cv2.imshow("mask",mask)	
		if(show):
			
			cv2.imshow("window", np.hstack([cv2.resize(image,(400,600)), cv2.resize(output,(400,600))]))
			cv2.waitKey(0)
	return images	

def visualizeBoundingBox(image_path, positions):
	image = cv2.imread(image_path)
	for position in positions:
		print position
		cv2.rectangle(image, (position[0], position[1]), (position[2], position[3]), (0, 255, 0), 2)
	cv2.imshow("images", image)
	cv2.waitKey(0)



# In[ ]:

def main( infer_ids, base_dir, save_dir):


    imgs = [cv2.imread(base_dir +'/' + 'segnet_vaihingen_128x128_64_area{}.tif'.format(l)) for l in infer_ids]
    #print imgs
    print "Processing {} images...".format(len(imgs))

    #predictions = [findcars(img,  step=64 , patch_size=(128, 128)) for img in imgs]

    bounding_cars = parseImage(imgs,boundaries[0],True)


# python find_vehicles 1 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegNet Inference Script')
    parser.add_argument('infer_ids', type=int, metavar='tiles', nargs='+',
                        help='id of tiles to be processed')
    parser.add_argument('--basedir', type=str,
                       help='Folder where to save the results')
    parser.add_argument('--dir', type=str,
                       help='Folder where to save the results')
    args = parser.parse_args()
    base_dir = args.basedir
    infer_ids = args.infer_ids
    save_dir = args.dir
    if save_dir is None:
        save_dir = './'
    if base_dir is None:
        base_dir = './'
    main(infer_ids, base_dir, save_dir)                   