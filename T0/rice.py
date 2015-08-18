# encoding: utf-8

"""Author: Pierre-Victor Chaumier <chaumierpv@gmail.com>
    
    Python Version : 3.4.2

    Required libraries :
        - Pillow        v2.9.0
        - scipy         v0.16.0
        - numpy         v1.9.2
        - matplotlib    v1.4.3

    To install them, simply run the command : pip install name-of-lib

    /!\ This script has only been tested with this version of Python and 
    /!\ with these versions of the libraries ! I give no garanty if you use
    /!\ a different set up.

    If you encounter an issue, please send me a mail to report it.

    NOTE : the use of a class to keep every image action together would improve
    the overall quality of this program.
"""

import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

from collections import Counter
from scipy import misc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


## -------------------- 0. General functions  -------------------

def plot_image(image, title, threshold=256):
    """Creates a new figure and display the image given in argument.

    The function also works with a 2D list.
    
    Note that it automatically increases the contrast. So if you give
    and image with grey values between 0 and 10 it will stretch the values
    to display the maximum of contrast (10 will appear to be 255 so white).
    On the opposite, an 2D list with values between 0 and 10000 will be
    compacted.
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
    plt.title(title)

## -------------------- 1. Image acquisition  -------------------

# the rice.png and this program must be in the same directory
# The rice.png image can easily be found on the internet.
rice = misc.imread('rice.png')
height, width = rice.shape


## -------------------- 2. Image display  -------------------

# Display the png image
plot_image(rice, 'Image in PNG')

# Display the 3D representation of the image (position and grey value)
# The level of grey are represented by temperature colors
fig = plt.figure(2)
ax = fig.gca(projection='3d')
X = np.arange(0, height, 1)
Y = np.arange(0, width, 1)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, rice, cmap=cm.coolwarm, linewidth=0)
plt.title('3D representation of image')


## -------------------- 3. Historiogram  -------------------

def freq_grey_level(image, number_of_grey):
    """Return an array of grey frequency. 

    Each index is a level of grey and the value is its frequency within the 
    original image.
    """

    freq = [0] * number_of_grey
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            freq[image[i][j]] += 1
    return freq

def display_grey_freq(image):
    """Display the graph grey value/frequency"""

    num_grey = 256
    grey_level = range(num_grey)
    freq = freq_grey_level(image, num_grey)
    plt.figure()
    plt.bar(grey_level, freq, color='blue')
    plt.xlabel('Grey level')
    plt.ylabel('Frequency')

# Actually displaying the graph
display_grey_freq(rice)

## -------------------- 4. Segmentation using a global  -------------------
## --------------------    threshold                    -------------------

def threshold_image(image, threshold):
    """Returns a numpy array thresholded.

    Note that the imshow include a threshold option but does not modify the
    given image. This function does.
    """
    image_tresholded = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_tresholded[i][j] = 0 if image[i][j] < threshold else 255
    return image_tresholded

# Here we only display the image thresholded without creating a 
# new one/modifying the old one.
plot_image(rice, 'Image with a threshold of 128', 128)
plot_image(rice, 'Image with a threshold of 150', 150)
plot_image(rice, 'Image with a threshold of 70', 70)


## -------------------- 5. Background elimination  -------------------

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

rice_bg_eliminated = background_elimination(rice)

plot_image(rice_bg_eliminated, 'Background elimination')


## -------------------- 6. Segmentation of new image  -------------------

# Following the same steps as in 3. and 4.
display_grey_freq(rice_bg_eliminated)
# Graph shows a limit around 40 so we use this value as a threshold
plot_image(rice_bg_eliminated, 'Image newly thresholded', 40)


## -------------------- 7. Elimination of small regions  -------------------
## -------------------- 8. Labeling of regions  -------------------

# I could not see a way to delete the small regions before marking them 
# So I started by marking the different regions and then I deleted the small 
# ones

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

    image_labelled = []
    for i in range(image.shape[0]):
        image_labelled.append([0] * image.shape[1])
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
                    
    # keep one example of each label class
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for el in equivalent_list:
                if image_labelled[i][j] in el:
                    image_labelled[i][j] = min(el)

    # Now number the region from 1 to 1
    new_labels = dict()
    current_new_label = 1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image_labelled[i][j] not in new_labels and \
               image_labelled[i][j] != 0:
                new_labels[image_labelled[i][j]] = current_new_label
                current_new_label += 1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image_labelled[i][j] != 0:
                image_labelled[i][j] = new_labels[image_labelled[i][j]]

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

rice_bg_eliminated_thresholded = threshold_image(rice_bg_eliminated, 40)
rice_labelled = label_region(rice_bg_eliminated_thresholded)
plot_image(rice_labelled, 'Rice labelled')
delete_small_region(rice_labelled, 10)
plot_image(rice_labelled, 'Rice labelled cleaned')

## --------------------  9. Feature extraction  -------------------



## --------------------  10. Classification of regions  -------------------

def frequency_size(image_labelled):
    """Display the graph frequency of size/size of regions."""
    c = determine_size_of_regions(image_labelled)
    # The biggest area is the black one, so we delete it
    del c[c.most_common(1)[0][0]]
    sizes = []
    for el in c:
        sizes.append(c[el])
    plt.figure()
    plt.hist(sizes)
    
    # This part is hard coded and thus will not work with other images
    plt.plot([100, 100], [0, 50], color='red', linestyle='--')
    plt.plot([300, 300], [0, 50], color='red', linestyle='--')
    plt.annotate('XS', xy=(50, 30))
    plt.annotate('OK', xy=(150, 30))
    plt.annotate('XL', xy=(400, 30))

    plt.xlabel('size of regions')
    plt.ylabel('Frequency')

frequency_size(rice_labelled)

def display_regions_size(image_labelled, min_size, max_size):
    """Display an image of the regions which size in pixels is between
    min_size and max_size."""
    copy_image = np.copy(image_labelled)
    c = determine_size_of_regions(image_labelled)
    for i, row in enumerate(image_labelled):
        for j, pixel in enumerate(row):
            if pixel != 0:
                if c[pixel] >= min_size and c[pixel] < max_size:
                    copy_image[i][j] = 255
                else:
                    copy_image[i][j] = 0
    return copy_image

## --------------------  11. Results  -------------------

rice_region_xs = display_regions_size(rice_labelled, 0, 70)
rice_region_ok = display_regions_size(rice_labelled, 71, 300)
rice_region_xl = display_regions_size(rice_labelled, 301, 1000)

plot_image(rice_region_xs, 'Regions XS')
plot_image(rice_region_ok, 'Regions M')
plot_image(rice_region_xl, 'Regions XL')

# The results are not exactly the same. I has to do with a different 
# implementation of the functions.

c = determine_size_of_regions(rice_labelled)
del c[c.most_common(1)[0][0]]
number_area_xs = 0
number_area_ok = 0
number_area_xl = 0
sizes = []
for el in c:
    sizes.append(c[el])
    if c[el] < 70:
        number_area_xs += 1
    elif c[el] < 300:
        number_area_ok += 1
    else:
        number_area_xl += 1

print('Statistics of OK rice:')
print('    >               Count:', number_area_ok)
print('    >        Average area:', stat.mean(sizes), ' [pixels]')
print('    >  Standard deviation:', stat.stdev(sizes), ' [pixels]')

## --------------------  End of Tarea 0  -------------------

# Display all the different figures
plt.show()

## --------------------  Tests  -------------------

# You can test the different functions here.

image_test = [
    [255, 0, 0, 0, 0, 0, 0, 0, 255],
    [0, 0, 0, 255, 0, 255, 0, 0, 0],
    [0, 0, 255, 255, 0, 255, 255, 0, 0],
    [0, 255, 255, 255, 0, 255, 255, 255, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 255, 255, 255, 0],
    [0, 0, 255, 255, 0, 255, 255, 0, 0],
    [0, 0, 0, 255, 0, 255, 0, 0, 0],
    [255, 0, 0, 0, 0, 0, 0, 0, 255]
]

image_test = np.array([np.array(line) for line in image_test])

# Represent the following image
# x = white, - = black
#        x-------x
#        ---x-x---
#        --xx-xx--
#        -xxx-xxx-
#        ---------
#        -xxx-xxx-
#        --xx-xx--
#        ---x-x---
#        x-------x

# image_test_labelled = label_region(image_test)
# plot_image(image_test_labelled, 'Image test labelled')
# delete_small_region(image_test_labelled, 1)
# plot_image(image_test_labelled, 'Image test cleaned')
# plt.show()
