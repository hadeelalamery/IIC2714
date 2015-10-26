import matplotlib.pyplot as plt
import numpy as np

import fct

from scipy import misc

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.color import gray2rgb

from skimage.feature import canny, peak_local_max
from skimage.morphology import dilation, disk
from skimage.transform import hough_circle


## --------------------  Images  -------------------

def read_img(nb, eye, img_nb):
    folder = 'input/IR/'
    file = folder + nb + '/' + nb + '_' + eye + '/'
    file += 'Img_' + nb + '_' + eye + '_' + img_nb + '.bmp'
    img = misc.imread(file)

    return img

img1 = read_img('001', 'L', '1')
img2 = read_img('002', 'L', '3')

## --------------------  Test  -------------------

img = gray2rgb(img1)

plt.figure()
segments = slic(img, n_segments=1000, sigma=5)
img_marked = mark_boundaries(img, segments)
plt.imshow(img_marked)

# plt.figure()
# segments = slic(img, n_segments=50, sigma=5, compactness=1)
# img_marked = mark_boundaries(img, segments)
# plt.imshow(img_marked)


## --------------------  Circle detection  -------------------

def circle_det(img, edges):
    hough_rad = np.arange(50, 150, 5)
    hough_res = hough_circle(edges, hough_rad)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_rad, hough_res):
        # For each radius, extract two circles
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    fig = plt.figure()
    image = color.gray2rgb(img)
    for idx in np.argsort(accums)[::-1][:5]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        circle = plt.Circle((center_y, center_x), radius, fill=False, color='r')
        fig.gca().add_artist(circle)
    plt.imshow(image)


## --------------------  Display images  -------------------

if True:
    plt.show(block=False)
    input('Hit enter to close')
    plt.close()
