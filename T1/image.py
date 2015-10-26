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
import statistics as stat

from collections import Counter
from scipy import misc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

## -------------------- Display image and characteristics  -------------------

def plot(image, title="", threshold=256):
    """Creates a new figure and display the image given in argument.

    The function also works with a 2D list.
    
    The threshold does not actually modify the given image. If you want to do
    so, you need to use the threshold function defined later.

    """

    plt.figure()
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

def draw_circle(fig, center, radius, color='red'):
    """Add a circle to current figure."""

    circle = plt.Circle(center, radius, fill=False, color=color)
    fig.gca().add_artist(circle)

def display_monochrome_3D(image):
    """Display the 3D representation of the image (position and gray value).

    The levels of gray are represented by temperature colors

    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, image.shape[0], 1)
    Y = np.arange(0, image.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, image, cmap=cm.coolwarm, linewidth=0)
    plt.title('3D representation of image')

def freq(image, nb_of_level=256, title='', xlabel='Level'):
    """Display the historiogram frequency/value."""

    plt.figure()
    plt.hist(image.flatten(), nb_of_level)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)

def freq_size(image_labelled):
    """Display the graph frequency of size/size of regions."""
    c = determine_size_of_regions(image_labelled)
    # The biggest area is the black one, so we delete it
    del c[c.most_common(1)[0][0]]
    sizes = []
    for el in c:
        sizes.append(c[el])
    plt.figure()
    plt.hist(sizes)

## -------------------- Transformation of image  -------------------

def rgb_2_gray(image):
    """Returns a new image in gray using the CIE RGB results.

        gray_level = 1.0000 * R + 4.5907 * G + 0.0601 * B
    
    https://en.wikipedia.org/wiki/CIE_1931_color_space
    The functions do not take into account the level of transparency.

    """

    img_gray = np.ndarray((image.shape[0], image.shape[1]))
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    img_gray[:, :] = (1.0000 * R + 4.5907 * G + 0.0601 * B) / 5.6508
    return img_gray


def rgb_2_monochrome(image, color):
    """Returns a new image in the color specified.

    The different possible color arguments are R, G or B.
    We assume we work on square images.

    """
    
    if color == 'R':
        return image[:, :, 0]
    elif color == 'G':
        return image[:, :, 1]
    elif color == 'B':
        return image[:, :, 2]


def threshold(image, threshold, threshold_end=255):
    """Returns a new numpy array which is a thresholded version of the given 
    one.

    Note that you can select an interval by specifying the threshold_end.

    /!\ The image given must be a monochrome (2D numpy array).
    
    """

    image_thresholded = np.copy(image)
    if threshold_end != 255:
        image_thresholded[
                (image_thresholded > threshold_end) |
                (image_thresholded < threshold)] = 0
        image_thresholded[
                (image_thresholded <= threshold_end) &
                (image_thresholded >= threshold)] = 255
    else:
        image_thresholded[image_thresholded < threshold] = 0
        image_thresholded[image_thresholded >= threshold] = 255
    return image_thresholded

## -------------------- Cleaning of image  -------------------

def background_elimination(image):
    """Returns an image with a normalized background.

    Works by going row by row and substracting the minimum of each row to each
    pixel of the row.

    This shifts the values towards the black.
    """
    image_bg_eliminated = np.copy(image)
    for i in range(image.shape[0]):
        mini = min(image[i])
        for j in range(image.shape[1]):
            image_bg_eliminated[i][j] = image[i][j] - mini
    return image_bg_eliminated


def add_equivalent_labels(equivalent_list, left_label, top_label):
    """Add an equivalence between two labels. 

    It also takes care of merging two subset that are equivent.
    """
    pos_left_label = -1
    pos_top_label = -1
    for i, el in enumerate(equivalent_list):
        if left_label in el:
            pos_left_label = i
        if top_label in el:
            pos_top_label = i
    # None of the two label already present
    if pos_left_label == -1 and pos_top_label == -1:
        equivalent_list.append(set([left_label, top_label]))
    # One of the two label is already in one sub-ensemble
    elif pos_left_label == -1 and pos_top_label != -1:
        equivalent_list[pos_top_label].add(left_label)
    elif pos_left_label != -1 and pos_top_label == -1:
        equivalent_list[pos_left_label].add(top_label)
    # Both are present but in different label sets
    elif pos_top_label != pos_left_label:
        equivalent_list[pos_top_label] = equivalent_list[pos_top_label].union(
                                             equivalent_list[pos_left_label])
        del equivalent_list[pos_left_label]

def renumber_region(image_labelled):
    # Now number the region from 1 to 1
    new_labels = dict()
    current_new_label = 1
    for i in range(image_labelled.shape[0]):
        for j in range(image_labelled.shape[1]):
            if image_labelled[i][j] not in new_labels and \
               image_labelled[i][j] != 0:
                new_labels[image_labelled[i][j]] = current_new_label
                current_new_label += 1
    for i in range(image_labelled.shape[0]):
        for j in range(image_labelled.shape[1]):
            if image_labelled[i][j] != 0:
                image_labelled[i][j] = new_labels[image_labelled[i][j]]

def label_region(image):
    """Labels all region in the binary image.

    The algorithm starts by checking if the current pixel is white.

    If so, it looks if the above and/or left pixel is white also
    and use their label to label the current pixel. If the top and left pixels
    are labelled but with a different label, it adds both the labels to an
    equivalent_list (which type is a list of sets).

    If they are not marked, it will create a new label and label it.

        - x -
        x o - 
        - - -

    o = current pixel
    x = top and left pixels
    - = pixels not considered
    """

    image_labelled = np.ndarray((image.shape[0], image.shape[1]))
    current_label = 0
    equivalent_list = []
    # i is the height and j the width
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # if the pixel is not black
            if image[i][j] != 0:
                # Edge cases of the position in the image
                # First pixel (0, 0)
                if i == 0 and j == 0:
                    current_label += 1
                    image_labelled[i][j] = current_label
                # On the first line (x, 0)
                elif i != 0 and j == 0:
                    if image[i - 1][j] != 0:
                        image_labelled[i][j] = image_labelled[i - 1][j]
                    else:
                        current_label += 1
                        image_labelled[i][j] = current_label
                # On the first column (0, x)
                elif i == 0 and j != 0:
                    if image[i][j - 1] != 0:
                        image_labelled[i][j] = image_labelled[i][j - 1]
                    else:
                        current_label += 1
                        image_labelled[i][j] = current_label
                else:
                    if image[i][j - 1] != 0:
                        # We use top label by default
                        image_labelled[i][j] = image_labelled[i][j - 1]
                        if image[i - 1][j] != 0 and \
                           image_labelled[i - 1][j] != image_labelled[i][j - 1]:
                            add_equivalent_labels(equivalent_list, 
                                                  image_labelled[i - 1][j],
                                                  image_labelled[i][j - 1])
                    elif image[i - 1][j] != 0:
                        image_labelled[i][j] = image_labelled[i - 1][j]
                    else:
                        current_label += 1
                        image_labelled[i][j] = current_label
                    
    renumber_region(image_labelled)

    # keep one example of each label class
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for el in equivalent_list:
                if image_labelled[i][j] in el:
                    image_labelled[i][j] = min(el)



    return image_labelled

def determine_size_of_regions(image_labelled):
    """Count how many pixel each regions has. Returns a Counter object."""
    c = Counter()
    for el in image_labelled:
        c += Counter(el)
    return c

def delete_small_region(image_labelled, size):
    """Remove areas of size inferior or equal to the size argument."""
    c = determine_size_of_regions(image_labelled)
    for i, row in enumerate(image_labelled):
        for j, pixel in enumerate(row):
            if pixel != 0:
                if c[pixel] <= size:
                    image_labelled[i][j] = 0
    renumber_region(image_labelled)


