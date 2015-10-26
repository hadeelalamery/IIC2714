# encoding: utf-8

"""Image processing Activity 2.

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

from skimage import color
from skimage.morphology import dilation, disk

from scipy import misc

## -------------------- Exercice 1: skin detection  -------------------

def ex1():
    img = misc.imread('people.bmp')

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    mask = (abs(R - G) + abs(R - B) + abs(G - B)) < 50
    img[mask] = (0, 0, 0)

    plt.figure()
    plt.imshow(img)

ex1()

## -------------------- Exercice 2: skin detection  -------------------

def ex2():
    img = misc.imread('gaviota.jpg')
    img2 = np.copy(img)

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    mask = np.logical_and(R < 40, G < 40)
    mask = dilation(mask, disk(2))

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            old_i = (i + 100) % mask.shape[0]
            old_j = (j + 50) % mask.shape[1]
            if mask[old_i][old_j]:
                img[i,j,:] = img[old_i,old_j,:]
    
    plt.figure()
    plt.imshow(img)

# ex2()

## -------------------- Exercice 3: change colors  -------------------

def ex3():
    img = misc.imread('iglesia.jpg')

    img2 = np.copy(img)
    plt.figure()
    plt.imshow(img2)

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    img[:,:,2] = 2 * img[:,:,2]

    plt.figure()
    plt.imshow(img)

# ex3()

## -------------------- Exercice 4: low contrast  -------------------

def ex4():
    img = misc.imread('lowcontrast.png')

    img2 = color.rgb2hsv(img[:,:,:3])
    H = img[:,:,0]
    S = img[:,:,1]
    V = img[:,:,2]

    plt.figure()
    plt.imshow(V)

    plt.figure()
    plt.hist(img2.flatten(), 256)


# ex4()

## -------------------- Plot images  -------------------

if True:
    plt.show(block=False)
    input('Hit enter to close')
    plt.close()
