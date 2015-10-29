# encoding: utf-8

"""Image processing library.

    author: Pierre-Victor Chaumier <chaumierpv@gmail.com>

    Python Version : 3.4.2

    Required libraries :
        - Pillow        v2.9.0
        - scipy         v0.16.0
        - numpy         v1.9.2
        - matplotlib    v1.4.3

    RQ : in all the fonctions, we assume that the images are rectangular.

"""

import matplotlib.pyplot as plt
import numpy as np

from scipy import misc

## -------------------- Display image and characteristics  -------------------

def display(image, title="", threshold=256):
    """Creates a new figure and display the image given in argument.

    The function also works with a 2D list.
    
    The threshold does not actually modify the given image. If you want to do
    so, you need to use the threshold function defined later.

    """

    height, width = image.shape
    plt.figure(figsize=(width/200, height/200))
    # There is a threshold included in the imshow function
    if threshold == 256:
        plt.imshow(image)
    else:
        plt.imshow(image > threshold)
    # pyplot uses thermal color by default, next line enables display
    # in black/white
    plt.gray()
    if title != "":
        plt.title(title)

def freq(image, nb_of_level=256, title='', xlabel='Level', cumulative=False):
    """Display the historiogram frequency/value."""

    fig = plt.figure()
    plt.hist(image.flatten(), nb_of_level, cumulative=cumulative)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)


## -------------------- Transformation of image  -------------------

def threshold(image, threshold, threshold_begin=0):
    """Returns a new numpy array which is a thresholded version of the given 
    one.

    Note that you can select an interval by specifying the threshold_end.

    /!\ The image given must be a monochrome (2D numpy array).
    
    """

    image_thresholded = np.copy(image)
    image_thresholded[:,:] = 0
    image_thresholded[(threshold_begin < image) & (image < threshold)] = 255
    return image_thresholded

def normalize(img):
    maxi = np.amax(img)
    mini = np.amin(img)
    return (255 * img / (maxi - mini) - (255 * mini / (maxi - mini)))
