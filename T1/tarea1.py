# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from scipy import misc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import image


## --------------------  1. Looking for the center  -------------------

def threshold_image(img):
    """We start by separating from white to black.

    As all images do not have the same repartition of level of grey
    we need to use a variating threshold.
    
    """
    img_gray = image.rgb_2_gray(img)
    threshold = np.mean(img_gray)
    img_gray_thresholded = image.threshold(img_gray, threshold - 15)
    return img_gray_thresholded

def clean_thresholded_image(img_thresholded, precision):
    """deletion of black column/rows (like the right column in the image3)."""
    for i in range(img_thresholded.shape[0]):
        if img_thresholded[i].sum() \
                <= 1 * precision * img_thresholded.shape[0]:
            img_thresholded[i, :] = 255
    for i in range(img_thresholded.shape[1]):
        if img_thresholded[:, i].sum() \
                <= 1 * precision * img_thresholded.shape[1]:
            img_thresholded[:, i] = 255


def find_center_radius(img_gray_thresholded, precision):
    """Returns a tuple with the coordonate of the center of the clock and the
    radius."""

    #
    # We try to find the first lines/rows that contains at least precision%
    # of black pixel from the left/right/top/down.
    #
    top = 0
    while top < img_gray_thresholded.shape[0] and \
          img_gray_thresholded[top].sum() \
                > 255 * precision * img_gray_thresholded.shape[0]:
        top += 1
    down = img_gray_thresholded.shape[0] - 1
    while down > 0 and \
          img_gray_thresholded[down].sum() \
            > 255 * precision * img_gray_thresholded.shape[0]:
        down -= 1
    left = 0
    while left < img_gray_thresholded.shape[1] and \
          img_gray_thresholded[:, left].sum() > 255 * precision * img_gray_thresholded.shape[1]:
        left += 1
    right = img_gray_thresholded.shape[1] - 1
    while right > 0 and \
          img_gray_thresholded[:, right].sum() > 255 * precision * img_gray_thresholded.shape[1]:
        right -= 1

    #
    #   Calculating the center and radius
    #

    center = ((top + down) / 2, (left + right) / 2)
    radius = (left - right) / 2
    
    #
    #   Display the centers
    #

    # # Uncomment following line to display
    # fig = plt.figure()
    # plt.imshow(img_gray_thresholded)
    # plt.gray()
    # plt.plot([center[1], center[1]], [0, img_gray_thresholded.shape[1]], color='red', linestyle='--')
    # plt.plot([0, img_gray_thresholded.shape[0]], [center[0], center[0]], color='red', linestyle='--')
    # plt.plot([left, left], [0, img_gray_thresholded.shape[1]], color='red', linestyle='--')
    # plt.plot([right, right], [0, img_gray_thresholded.shape[1]], color='red', linestyle='--')
    # plt.plot([0, img_gray_thresholded.shape[0]], [top, top], color='red', linestyle='--')
    # plt.plot([0, img_gray_thresholded.shape[0]], [down, down], color='red', linestyle='--')
    
    return center, radius


## --------------------  Finding the time  -------------------

def only_red(img):
    """Transform the image to keep the parts that contains two times more
    red than green.
    
    As the green and blue levels are equal in the images given, 
    it is not necessary to use red > green + blue to threshold.

    """
    
    red = np.ndarray((img.shape[0], img.shape[1]))
    red.fill(0)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] > 2 * img[i][j][1]:
                red[i][j] = 255
    return red

def angle_to_unit(avg_angle, nb_of_unit):
    """Transform a given angle to a unit number."""

    unit_float = nb_of_unit * avg_angle / (2 * np.pi)
    unit = int(unit_float)
    if unit_float - unit > 0.5:
        return unit + 1
    else:
        return unit

def time(img, center, radius):
    """Find the time given the image of a clock, its center and radius."""

    time = [0, 0, 0]

    #
    #   Defining the circles for detection.
    #

    # 600 tetas
    teta = np.arange(0, 2 * np.pi, np.pi / 300)
    # Small circle
    x_s_circle = (radius / 2) * np.cos(teta) + center[0]
    y_s_circle = (radius / 2) * np.sin(teta) + center[1]
    # Large circle
    x_l_circle = (radius / 1.5) * np.cos(teta) + center[0]
    y_l_circle = (radius / 1.5) * np.sin(teta) + center[1]

    #
    #   Detect seconds
    #

    begin_sec = 0
    end_sec = 0
    i = 0

    # Detect first and last pixel
    img_only_red = only_red(img)

    if img_only_red[x_l_circle[i]][y_l_circle[i]] == 0:
        while i < len(x_l_circle) and \
              img_only_red[x_l_circle[i]][y_l_circle[i]] == 0:
            i += 1
        begin_sec = i
        end_sec = begin_sec
        while i < len(x_l_circle) and \
              img_only_red[x_l_circle[i]][y_l_circle[i]] != 0:
            i += 1
        end_sec = i

    # Changing angle to secs
    avg_sec = int((end_sec + begin_sec) / 2)
    teta_sec_avg = 2 * np.pi - teta[avg_sec]
    time[2] = angle_to_unit(teta_sec_avg, 60)

    #
    #   Detect minutes
    #

    begin_min = 0
    end_min = 0
    i = 0

    img_gray_thresholded = threshold_image(img)

    # Changin begin_sec and end_sec to correspond to the thresholded graph
    end_sec = (begin_sec + end_sec) // 2
    while img_gray_thresholded[x_l_circle[end_sec]][y_l_circle[end_sec]] == 0:
        end_sec += 1
    begin_sec = (begin_sec + end_sec) // 2
    while img_gray_thresholded[x_l_circle[begin_sec]][y_l_circle[begin_sec]] == 0:
        begin_sec -= 1

    # Detecting begin and end of minute bar
    if img_gray_thresholded[x_l_circle[i]][y_l_circle[i]] != 0:
        while i < len(x_l_circle) and \
              (img_gray_thresholded[x_l_circle[i]][y_l_circle[i]] != 0 or
              i in range(begin_sec, end_sec + 1)):
            i += 1
        begin_min = i
        # Test if the sec and minute bar are not superimposed
        if begin_min != 600:
            end_min = begin_min
            while i < len(x_l_circle) and \
                  img_gray_thresholded[x_l_circle[i]][y_l_circle[i]] == 0:
                i += 1
            end_min = i

            # Changing angle to min
            avg_min = int((end_min + begin_min) / 2)
            teta_min_avg = 2 * np.pi - teta[avg_min]
            time[1] = angle_to_unit(teta_min_avg, 60)
        else:
            begin_min, end_min = begin_sec, end_sec
            avg_min = int((end_min + begin_min) / 2)
            avg_min = 2 * avg_min - avg_sec
            teta_min_avg = 2 * np.pi - teta[avg_min]
            time[1] = angle_to_unit(teta_min_avg, 60)

    #
    #   Detect hours
    #

    begin_hour = 0
    end_hour = 0
    i = 0

    # Changin begin_sec, end_sec, begin_min and end_min to correspond
    # to the thresholded graph and the small circle.
    end_sec = (begin_sec + end_sec) // 2
    while img_gray_thresholded[x_s_circle[end_sec]][y_s_circle[end_sec]] == 0:
        end_sec += 1
    begin_sec = (begin_sec + end_sec) // 2
    while img_gray_thresholded[x_s_circle[begin_sec]][y_s_circle[begin_sec]] == 0:
        begin_sec -= 1

    end_min = (begin_min + end_min) // 2
    while img_gray_thresholded[x_s_circle[end_min]][y_s_circle[end_min]] == 0:
        end_min += 1
    begin_min = (begin_min + end_min) // 2
    while img_gray_thresholded[x_s_circle[begin_min]][y_s_circle[begin_min]] == 0:
        begin_min -= 1

    # Detecting hour bar
    if img_gray_thresholded[x_s_circle[i]][y_s_circle[i]] != 0:
        while i < len(x_s_circle) and \
              (img_gray_thresholded[x_s_circle[i]][y_s_circle[i]] != 0 or
              i in range(begin_sec, end_sec) or
              i in range(begin_min, end_min)):
            i += 1
        begin_hour = i
        # Test if the hour bar is not on top of minute and/or sec bar.
        if begin_hour != 600:
            end_hour = begin_hour
            while i < len(x_s_circle) and \
                  img_gray_thresholded[x_s_circle[i]][y_s_circle[i]] == 0:
                i += 1
            end_hour = i
        
            # Changing angle to hours
            avg = int((end_hour + begin_hour) / 2)
            teta_hour_avg = 2 * np.pi - teta[avg]
            time[0] = int(12 * teta_hour_avg / (2 * np.pi))

        else:
            # We compare the distance between the two detected begining and
            # ending points for sec and min.
            # The distance separating the sec should be inferior to the one
            # separating the min. If not, the hour bar is on the sec bar.
            d_sec = (y_s_circle[begin_sec] - y_s_circle[end_sec]) ** 2 + \
                    (x_s_circle[begin_sec] - x_s_circle[end_sec]) ** 2

            d_min = (y_s_circle[begin_min] - y_s_circle[end_min]) ** 2 + \
                    (x_s_circle[begin_min] - x_s_circle[end_min]) ** 2

            if d_sec > 0.5 * d_min:
                begin_hour, end_hour = begin_sec, end_sec
                avg_hour = int((end_hour + begin_hour) / 2)
                avg_hour = 2 * avg_hour - avg_sec
                teta_hour_avg = 2 * np.pi - teta[avg_hour]
                time[0] = int(12 * teta_hour_avg / (2 * np.pi))
            else:
                begin_hour, end_hour = begin_min, end_min
                avg_hour = int((end_hour + begin_hour) / 2)
                avg_hour = 2 * avg_hour - avg_min
                teta_hour_avg = 2 * np.pi - teta[avg_hour]
                time[0] = int(12 * teta_hour_avg / (2 * np.pi))


    #
    #   Display the detected pixels
    #

    # # Uncomment the following lines to see where the different points are.
    # fig = plt.figure()
    # plt.imshow(img_gray_thresholded)
    # plt.plot(y_l_circle, x_l_circle, color='black')
    # plt.plot(y_s_circle, x_s_circle, color='black')
    # plt.plot([center[1], center[1]], [0, img_gray_thresholded.shape[1]], color='black', linestyle='--')
    # plt.plot([0, img_gray_thresholded.shape[0]], [center[0], center[0]], color='black', linestyle='--')
    # plt.gray()
    # radius = 5

    # center = y_s_circle[begin_sec], x_s_circle[begin_sec]
    # image.draw_circle(fig, center, radius, 'green')
    # center = y_s_circle[end_sec], x_s_circle[end_sec]
    # image.draw_circle(fig, center, radius, 'green')

    # center = y_s_circle[begin_min], x_s_circle[begin_min]
    # image.draw_circle(fig, center, radius, 'blue')
    # center = y_s_circle[end_min], x_s_circle[end_min]
    # image.draw_circle(fig, center, radius, 'blue')

    # center = y_s_circle[begin_hour], x_s_circle[begin_hour]
    # image.draw_circle(fig, center, radius, 'red')
    # center = y_s_circle[end_hour], x_s_circle[end_hour]
    # image.draw_circle(fig, center, radius, 'red')

    return time

## --------------------  Main Program  -------------------

precision = 0.95

def main():
    """Main program"""
    img_path = input('Give the path to your image :\n')
    img = misc.imread(img_path)
    img_gray_thresholded = threshold_image(img)
    clean_thresholded_image(img_gray_thresholded, precision)
    center, radius = find_center_radius(img_gray_thresholded, precision)
    t = time(img, center, radius)
    time_str = str(t[0]) + 'h' + str(t[1]) + 'm' + str(t[2]) + 's'
    print('Hello Mister Mery, the time is', time_str)

main()

## --------------------  Test  -------------------

# img1 = misc.imread('clock2015/clock_01.png')
# img2 = misc.imread('clock2015/clock_02.png')
# img3 = misc.imread('clock2015/clock_03.png')
# img4 = misc.imread('clock2015/clock_04.png')
# img5 = misc.imread('clock2015/clock_05.png')
# img6 = misc.imread('clock2015/clock_06.png')
# img7 = misc.imread('clock2015/clock_07.png')

# img_gray_thresholded1 = threshold_image(img1)
# img_gray_thresholded2 = threshold_image(img2)
# img_gray_thresholded3 = threshold_image(img3)
# img_gray_thresholded4 = threshold_image(img4)
# img_gray_thresholded5 = threshold_image(img5)
# img_gray_thresholded6 = threshold_image(img6)
# img_gray_thresholded7 = threshold_image(img7)

# clean_thresholded_image(img_gray_thresholded1, precision)
# clean_thresholded_image(img_gray_thresholded2, precision)
# clean_thresholded_image(img_gray_thresholded3, precision)
# clean_thresholded_image(img_gray_thresholded4, precision)
# clean_thresholded_image(img_gray_thresholded5, precision)
# clean_thresholded_image(img_gray_thresholded6, precision)
# clean_thresholded_image(img_gray_thresholded7, precision)

# center1, radius1 = find_center_radius(img_gray_thresholded1, precision)
# print(time(img1, center1, radius1))
# plt.savefig('figures/center/center_img1')
# center2, radius2 = find_center_radius(img_gray_thresholded2, precision)
# print(time(img2, center2, radius2))
# plt.savefig('figures/center/center_img2')
# center3, radius3 = find_center_radius(img_gray_thresholded3, precision)
# print(time(img3, center3, radius3))
# plt.savefig('figures/center/center_img3')
# center4, radius4 = find_center_radius(img_gray_thresholded4, precision)
# print(time(img4, center4, radius4))
# plt.savefig('figures/center/center_img4')
# center5, radius5 = find_center_radius(img_gray_thresholded5, precision)
# print(time(img5, center5, radius5))
# plt.savefig('figures/center/center_img5')
# center6, radius6 = find_center_radius(img_gray_thresholded6, precision)
# print(time(img6, center6, radius6))
# plt.savefig('figures/center/center_img6')
# center7, radius7 = find_center_radius(img_gray_thresholded7, precision)
# print(time(img7, center7, radius7))
# plt.savefig('figures/center/center_img7')


# # Display the plots, press enter to close all of them at once
# plt.show(block=False)
# input("Hit Enter To Close")
# plt.close()


## --------------------  End of Homework  -------------------
