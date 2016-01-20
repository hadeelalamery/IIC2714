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

from collections import Counter
from scipy import misc
from sklearn import cluster

from skimage.measure import label
from skimage.morphology import remove_small_objects, dilation, disk
from skimage.exposure import equalize_adapthist
from skimage.feature import canny

# Clean iris
from skimage.morphology import (convex_hull_image, binary_erosion, 
                                dilation, erosion)

# Find center
from scipy.signal import convolve2d as conv2


## -------------------- General functions  -------------------

def read_img(nb, eye, img_nb):
    """Read from the image folder."""

    folder = 'input/IR/'
    file = folder + nb + '/' + nb + '_' + eye + '/'
    file += 'Img_' + nb + '_' + eye + '_' + img_nb + '.bmp'
    img = misc.imread(file)

    return img

def kmeans(img, n_clusters=10):
    """Kmeans clustering technique for b&w images."""

    np.random.seed(0)

    X = img.reshape((-1, 1))  # We need an (n_sample, n_feature) array
    k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_

    # create an array from labels and values
    img_compressed = np.choose(labels, values)
    img_compressed.shape = img.shape

    return img_compressed


## -------------------- Pupil detection  -------------------

def segment_lowest_cluster(img_k):
    """Returns a binarized image with only the smallest 
    cluster of the kmeans."""

    mini_img_k = np.amin(img_k)
    darkest_cluster = dilation(img_k < mini_img_k + 1, disk(3))
    return darkest_cluster


def cluster_mean(darkest_cluster):
    """Get the mean of the cluster."""

    X, Y = [], []
    for i in range(darkest_cluster.shape[0]):
        for j in range(darkest_cluster.shape[1]):
            if darkest_cluster[i][j]:
                X.append(i)
                Y.append(j)
    return np.mean(X), np.mean(Y)


def find_pupil(darkest_cluster):
    """Find the pupil area by cleaning the darkest cluster."""

    x, y = cluster_mean(darkest_cluster)
    
    # Find pupil region
    cleaned_pupil = np.copy(darkest_cluster)
    labels = label(darkest_cluster)
    c = Counter(labels.flatten())
    label_pupil = c.most_common(2)[1][0]
    clean_pupil = labels == label_pupil
    
    # Naive radius as the half distance between 
    # the most right element and most left element
    # of the pupil region
    sums = np.sum(clean_pupil, axis=1)
    radius = np.amax(sums) / 2

    # Delete areas that are not in the naive radius of the iris
    for i in range(cleaned_pupil.shape[0]):
        for j in range(cleaned_pupil.shape[1]):
            if np.linalg.norm([x - i, y - j]) > 1.1 * radius:
                cleaned_pupil[i,j] = 0
                
    # Add again the region labelled as the pupil before
    cleaned_pupil[clean_pupil] += 1

    # Find new naive center
    sums = np.sum(clean_pupil, axis=1)
    radius = np.amax(sums) / 2
    
    # Fill using a convex hull the pupil
    chull = convex_hull_image(cleaned_pupil)
    cleaned_pupil[chull] += 1
    
    return cleaned_pupil, radius


def center(clean_pupil, radius):
    """Center is the point of max conv with a disk."""
    
    # Find the small region that contains the pupil
    left, right, up, down = 0, 0, 0, 0
    for i in range(clean_pupil.shape[0]):
        for j in range(clean_pupil.shape[1]):
            if clean_pupil[i,j] != 0:
                left = min(i, left)
                up = min(j, up)
                right = max(i, right)
                down = max(j, down)
    left = max(left - radius, 0)
    right = min(right + radius, clean_pupil.shape[0] - 1)
    up = max(up - radius, 0)
    down = min(down + radius, clean_pupil.shape[1] - 1)

    small_pupil = clean_pupil[left:right, up:down]

    # Convolution of cleaned pupil with a disk
    cp = np.zeros(small_pupil.shape)
    cp[small_pupil] = 1
    d = disk(radius)
    t4 = conv2(cp, d, mode='same')
    
    # Find center as the max of the convolution
    # product
    maxi = np.amax(t4)
    Y = []
    most_left, most_right = clean_pupil.shape[0], 0
    for i in range(t4.shape[0]):
        for j in range(t4.shape[1]):
            if t4[i,j] == maxi:
                Y.append(j)
            if clean_pupil[i,j] != 0:
                most_left = min(most_left, i)
                most_right = max(most_left, i)

    y_max = np.mean(Y)
    y_max += up
    x_max = (most_right - most_left) // 2 + most_left
    
    return x_max, y_max


def radius(pupil, center_x, center_y):
    """Radius as the distnace"""

    edges = canny(pupil)
    distances = []
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i,j] != 0:
                dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                # Keep only edges that are between two circles defined by
                # two times the radius and 350
                distances.append(dist)

    hist = np.histogram(distances, bins=50)

    for i in range(len(hist[0])):
        if hist[0][i] > 20:
            rad = hist[1][i]

#    plt.figure()
#    plt.hist(distances, bins=50)
#    plt.show()
    
    return rad


## -------------------- Iris detection  -------------------

def bin_image(img):

    it = np.copy(img)
    it_ad = equalize_adapthist(it)

    # Kmeans to label different area
    nb_clusters = 20
    itk = kmeans(it, nb_clusters)
    itk_ad = kmeans(it_ad, nb_clusters)

    # Keep only iris and pupil
    nb_clusters_to_keep = 7
    t_ad = list(set(itk_ad.flatten()))
    t_ad.sort()
    threshold = t_ad[nb_clusters_to_keep]
    ii_ad = itk_ad <= threshold
    ii_ad = dilation(erosion(ii_ad, disk(5)), disk(5))
    
    return ii_ad

def radius_iris(ii_ad_canny, center_x, center_y, rad, plot_hist=False):

    distances = []
    for i in range(ii_ad_canny.shape[0]):
        for j in range(ii_ad_canny.shape[1]):
            if i > center_x and ii_ad_canny[i,j] != 0:
                dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                # Keep only edges that are between two circles defined by
                # two times the radius and 350
                if 2 * rad < dist < 350:
                    distances.append(dist)
    
    hist = np.histogram(distances, bins=50)
    if plot_hist:
        plt.hist(distances, bins=50)
        plt.show()

    return hist[1][np.argmax(hist[0])]

def plot_iris_edge(ii_ad_canny, center_x, center_y, rad):
    # Plot original image with limits calculated with a canny
    plt.imshow(ii_ad_canny)
    X = []
    Y = []
    for i in range(ii_ad_canny.shape[0]):
        for j in range(ii_ad_canny.shape[1]):
            dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if 2 * rad < dist < 350 and i > center_x and ii_ad_canny[i,j] != 0:
                X.append(i)
                Y.append(j)
    plt.scatter(Y, X, color='r')
    plt.scatter(center_y, center_x, color='r')
