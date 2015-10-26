# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from scipy import misc

import fct

def low_filter(img, d):
    N = len(img)
    mask = [1] * (d + 1) + [0] * (N - 2 * d - 1) + [1] * d
    A = np.copy(img)
    B = np.copy(img)
    for i in range(N):
        A[i,:] = np.fft.ifft(np.fft.fft(img[i,:]) * mask)
        B[:,i] = np.fft.ifft(np.fft.fft(img[:,i]) * mask)
    return A / 2 + B / 2
    
file = 'zebra_noise.png'
img = misc.imread(file)

plt.figure()
plt.gray()
plt.imshow(img)

plt.figure()
for i in range(1, 10):
    ax = plt.subplot(3,3,i)
    ax.set_title('Filter : ' + str(i * 10))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    AB = low_filter(img, 10 * i)
    plt.imshow(AB)

plt.show(block=False)
input("Hit Enter To Close")
plt.close()
