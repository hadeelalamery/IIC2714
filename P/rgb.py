import matplotlib.pyplot as plt
import numpy as np

import fct

from scipy import misc
from skimage.morphology import dilation, disk

from skimage.segmentation import slic, mark_boundaries
from skimage.feature import canny, peak_local_max
from skimage.transform import hough_circle
from skimage import color


## --------------------  Images  -------------------

def read_img(nb, eye, img_nb):
    folder = 'input/RGB/'
    file = folder + nb + '/'
    file += 'IMG_' + nb + '_' + eye + '_' + img_nb + '.JPG'
    img = misc.imread(file)

    return img

img = read_img('001', 'L', '1')

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

plt.imshow(img)

plt.gray()
segments = slic(img, n_segments=1000, sigma=20)
img_marked = mark_boundaries(img, segments)
edges = canny(fct.normalize(segments))
edges = dilation(edges, disk(5))

hough_rad = np.arange(500, 1000, 20)
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
image = img_marked
for idx in np.argsort(accums)[::-1][:5]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    circle = plt.Circle((center_y, center_x), radius, fill=False, color='r')
    fig.gca().add_artist(circle)
plt.imshow(image)

# fct.freq(R)
# fct.freq(G)
# fct.freq(B)

# mask = R < 150
# mask = dilation(mask, disk(10))
# img[mask] = (0, 0, 0)

## --------------------  Display images  -------------------

if True:
    plt.show(block=False)
    input('Hit enter to close')
    plt.close()
