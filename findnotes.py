#!/usr/bin/env python2

# Copyright 2014 Maurice van der Pot
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import copy
import cv2
from datetime import datetime as dt
import json
import math
import numpy as np
import os
import sys

import common
import webcam

from PyQt5 import QtWidgets

def calculate_parameters_from_mat(stack_of_y):
    weight = 1.0 / len(stack_of_y)
    yshape = stack_of_y[0].shape

    mean_y = np.zeros(yshape, dtype=np.float32)
    for y in stack_of_y:
        mean_y = cv2.addWeighted(mean_y, 1.0, y, weight, 0)

    mean_corrected_y = [y - mean_y for y in stack_of_y]

    element_wise_y_squared = [np.float32(y) * y for y in mean_corrected_y]
    mean_corrected_y_squared = np.zeros(yshape, dtype=np.float32)
    for y_squared in element_wise_y_squared:
        mean_corrected_y_squared = cv2.add(mean_corrected_y_squared, y_squared)

    mean_corrected_y_norm = cv2.sqrt(mean_corrected_y_squared)

    return mean_y, mean_corrected_y, mean_corrected_y_squared, mean_corrected_y_norm

def createCircularKernel():
    diameter = common.NOTE_SIZE
    circleKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter, diameter))
    return circleKernel

def determine_average_colors(image, circleKernel):
    kernelSize = cv2.countNonZero(circleKernel)

    averages = cv2.filter2D(image, -1, circleKernel.astype(np.float32) / kernelSize)
    return averages

def normalized(img):
    norm = np.zeros(img.shape, dtype=np.float32)
    norm = cv2.normalize(img, norm, 0, 1, cv2.NORM_MINMAX)
    return np.float32(norm)

def filter_insignificant_hue(hsv):
    h, s, v = cv2.split(hsv)
    h[s < 25] = 0
    hsv = cv2.merge( (h, s, v) )
    return hsv

def findnotes(image):
    #averages = determine_average_colors(image)

    #common.qimshow(image)

    median = cv2.medianBlur(image, 11)
    #common.qimshow(median)
    medianlab = cv2.cvtColor(median, cv2.COLOR_BGR2Lab)

    different_color_threshold = 31

    common.qimshow(image)


    circle = createCircularKernel()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)
    retval, color = cv2.threshold(saturation, 25, 1, cv2.THRESH_BINARY)

    sum_of_color_in_circle = cv2.filter2D(color, cv2.CV_32F, circle)
    #common.qimshow([normalized(circle), color, normalized(filtered_color)])

    circleArea = cv2.countNonZero(circle)
    retval, significant_color_mask = cv2.threshold(sum_of_color_in_circle, 2 * circleArea / 3, 255, cv2.THRESH_BINARY)
    significant_color_mask = significant_color_mask.astype(np.uint8)
    minval, maxval, _, _ = cv2.minMaxLoc(sum_of_color_in_circle)
    print 'minval = %f, maxval = %f' % (minval, maxval)
    #common.qimshow([color, normalized(sum_of_color_in_circle), normalized(significant_color_mask)])

    kernel = np.ones((common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), np.uint8)
    color_dilated = cv2.dilate(sum_of_color_in_circle, kernel)
    local_max_at_zero = np.float32(sum_of_color_in_circle) - color_dilated
    local_max_mask = cv2.inRange(local_max_at_zero, -1.0, 1.0)

    max_color_mask = cv2.bitwise_and(local_max_mask, significant_color_mask)

    local_max_image = image.copy()
    local_max_image[local_max_mask == 255] = (255,0,255)

    significant_color_image = image.copy()
    significant_color_image[significant_color_mask == 0] = (0,0,0)

    max_color_image = image.copy()
    max_color_image[max_color_mask == 255] = (255,0,255)

    _, contours, _ = cv2.findContours(max_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotatedimage = image.copy()
    for contour in contours:
        center, _ = cv2.minEnclosingCircle(contour)
        cv2.circle(annotatedimage, tuple(int(f) for f in center), common.NOTE_SIZE / 2, (255,0,255))

    '''
    common.qimshow([ ['saturation', saturation],
                     #[sum_of_color_in_circle, normalized(sum_of_color_in_circle)],
                     #[color_dilated, normalized(color_dilated)],
                     #[local_max_at_zero, normalized(local_max_at_zero)],
                     ['local max', local_max_mask, local_max_image],
                     ['significant color', significant_color_mask, significant_color_image],
                     ['max color', max_color_mask, max_color_image],
                     ['found notes', annotatedimage] ])
    '''





    '''
    # 1-channel mode
    padsize = common.NOTE_SIZE / 2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    v[saturation < 25] = 0
    
    paddedv = np.pad(v, ((padsize, padsize), (padsize, padsize)), 'constant')

    circle_indices = np.where(circle != 0)

    mode_filtered = cv2.cvtColor(np.zeros(image.shape, dtype=np.uint8), cv2.COLOR_BGR2HSV)
    _, _, mode_filtered = cv2.split(mode_filtered)

    mode_count = np.zeros(image.shape[:2], dtype=np.float32)
    color_indices = np.where(significant_color_mask != 0)
    for y_offset, x_offset in zip(*color_indices):
        y_indices = circle_indices[0] + y_offset
        x_indices = circle_indices[1] + x_offset

        colors = paddedv[(y_indices, x_indices)]
        values, counts = np.unique(colors, return_counts=True)
        mode = values[np.argmax(counts)]
        mode_filtered[y_offset][x_offset] = mode

        absdiff = np.absolute(np.float32(values) - mode)
        count = np.sum(counts[absdiff < different_color_threshold])

        #if mode == 160:
        #    print 'unique     ', values
        #    print 'occurrences', counts
        #    print 'total      ', len(counts)
        #    print 'withinlimit', count

        mode_count[y_offset][x_offset] = count

    minval, maxval, _, _ = cv2.minMaxLoc(mode_count, significant_color_mask)
    print 'mode count ranges from %f to %f' % (minval, maxval)
    mode_count[significant_color_mask == 0] = minval
    common.qimshow([ ['paddedv', paddedv],
                     ['mode_filtered', mode_filtered],
                     ['mode_count', normalized(mode_count)] ])
    '''


    before = dt.now()

    # 3-channel mode
    padsize = common.NOTE_SIZE / 2

    color_absence = np.float32((1 - color) * different_color_threshold)
    #color_absence = np.float32((1 - color))

    padded = np.pad(image, ((padsize, padsize), (padsize, padsize), (0,0)), 'constant')
    paddedlab = cv2.cvtColor(padded, cv2.COLOR_BGR2Lab)
    color_absence_padded = np.pad(color_absence, ((padsize, padsize), (padsize, padsize)), 'constant', constant_values = different_color_threshold)

    h, w, c = padded.shape
    paddedmerged = np.zeros((h, w, c + 1), dtype=np.uint8)
    paddedmerged[:,:,:3] = paddedlab
    paddedmerged[:,:,3] = color_absence_padded
    paddedpacked = paddedmerged.view(np.uint32)

    my_x = 431 - 350 #417 - 350
    my_y = 716 #616 - 550

    circle_indices = np.where(circle != 0)

    mode_filtered = cv2.cvtColor(np.zeros(image.shape, dtype=np.uint8), cv2.COLOR_BGR2Lab)

    mode_count = np.zeros(image.shape[:2], dtype=np.float32)
    color_indices = np.where(significant_color_mask != 0)
    for y_offset, x_offset in zip(*color_indices):
        y_indices = circle_indices[0] + y_offset
        x_indices = circle_indices[1] + x_offset

        colors = paddedpacked[(y_indices, x_indices)]
        values, counts = np.unique(colors, return_counts=True)
        mode_packed = values[np.argmax(counts)]

        mode = np.array([mode_packed], dtype=np.uint32).view(np.uint8)
        lab_values = values.reshape((values.shape[0], 1)).view(np.uint8)
        absdiff = np.absolute(np.float32(lab_values) - mode)

        mode_filtered[y_offset][x_offset] = mode[:3]

        eucl_diffs = np.sqrt(np.sum(absdiff * absdiff, axis=1))

        if x_offset == my_x and y_offset == my_y:

            print 'here we go'
            print 'mode at (%d,%d) is %s' % (my_x, my_y, mode)

            for count, value, diff in zip(counts[eucl_diffs <= different_color_threshold], lab_values[eucl_diffs <= different_color_threshold], eucl_diffs[eucl_diffs <= different_color_threshold]):
                if value[3] > 20:
                    print '%d matching pixels of value (%s) had distance of %f' % (count, value, diff)

        count = np.sum(counts[eucl_diffs <= different_color_threshold])

        mode_count[y_offset][x_offset] = count

        if x_offset == my_x and y_offset == my_y:

            print 'mode_count', count
            print 'absdiff', absdiff
            print 'nr of circle pixels', len(circle_indices[0])

            dinges = paddedpacked.copy()
            for value in values[eucl_diffs <= different_color_threshold]:
                dinges[paddedpacked == value] = 0
            dinges2 = dinges.view(np.uint8)
            dinges3 = np.zeros((dinges2.shape[0], dinges2.shape[1], dinges2.shape[2] - 1), dtype=np.uint8)
            dinges3[:,:,:] = dinges2[:,:,:3]
            dinges3= cv2.cvtColor(dinges3, cv2.COLOR_Lab2BGR)
            common.qimshow(dinges3)


            print 'mode_count', count


    minval, maxval, _, _ = cv2.minMaxLoc(mode_count)
    print 'mode_count ranges from %f to %f' % (minval, maxval)

    after = dt.now()

    print after - before

    common.qimshow(normalized(mode_count))

    '''
    print 'mode at (%d,%d) is %s' % (my_x, my_y, mode_filtered[my_y][my_x])

    imagelab = image.copy()
    #imagelab[significant_color_mask == 0] = (0,0,0)
    imagelab = cv2.cvtColor(imagelab, cv2.COLOR_BGR2Lab)
    mode_count2 = np.zeros(image.shape[:2], dtype=np.float32)


    for y in range(common.NOTE_SIZE / 2, image.shape[0] - common.NOTE_SIZE / 2):
        for x in range(common.NOTE_SIZE / 2, image.shape[1] - common.NOTE_SIZE / 2):
            if significant_color_mask[y][x] == 0:
                continue
            one_mode = mode_filtered.copy()
            one_mode[:] = one_mode[y][x]

            absdiff = cv2.absdiff(imagelab, one_mode)
            b, g, r = cv2.split(absdiff)
            color_absence = np.float32((1 - color) * different_color_threshold)
            norm = cv2.sqrt(np.float32(b) * b + np.float32(g) * g + np.float32(r) * r + color_absence * color_absence)

            circle_indices = np.where(circle != 0)
            x_indices = circle_indices[1] + x - common.NOTE_SIZE / 2
            y_indices = circle_indices[0] + y - common.NOTE_SIZE / 2

            outsidecircle = np.ones(mode_count2.shape, np.uint8)
            outsidecircle[(y_indices, x_indices)] = 0

            matchingpixels = np.zeros(mode_count2.shape, np.uint8)
            matchingpixels[norm <= different_color_threshold] = 255
            matchingpixels[outsidecircle == 1] = 0

            if y == my_y and x == my_x:

                import itertools
                occurrences = {}
                tuples = []
                for rgb, dist in zip(imagelab[matchingpixels == 255], norm[matchingpixels == 255]):
                    trgb = tuple(rgb)
                    tuples.append(trgb)
                    occurrences[trgb] = dist

                for value, count in sorted([(g[0], len(list(g[1]))) for g in itertools.groupby(sorted(tuples))]):
                    if count > 3:
                        print '%d matching pixels of value (%d,%d,%d) had distance of %f' % (count, value[0], value[1], value[2], occurrences[value])

            mode_count2[y][x] = cv2.countNonZero(matchingpixels)

    print 'mode_count at (%d,%d) is %f, while mode_count2 is %f' % (my_x, my_y, mode_count[my_y][my_x], mode_count2[my_y][my_x])
    common.qimshow([ ['real mode_count', mode_count / 2000],
                     ['slow mode_count', mode_count2 / 2000] ])
    '''


    '''
    # specific positions
    for x, y in [ (412, 621), (417, 616) ]:
    #for x, y in [ (48, 749), (35, 755), (33, 740), (44, 739), (56, 742), (54, 759) ]:
    #for x, y in [ (441, 756), (1220, 587), (580, 644), (467, 756), (44, 750) ]:
    #for x, y in [ (346, 165), (234, 100), (344, 146) ]:

        x -= 350
        y -= 550

        #one_mode = cv2.cvtColor(mode_filtered, cv2.COLOR_Lab2BGR)
        #one_mode = cv2.cvtColor(one_mode, cv2.COLOR_BGR2HSV)
        one_mode = mode_filtered.copy()
        one_mode[:] = one_mode[y][x]

        imagelab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        imagelabcopy = imagelab.copy()
        cv2.circle(imagelabcopy, (x,y), 0, (255,0,0))

        mode_filtered_copy = mode_filtered.copy()
        cv2.circle(mode_filtered_copy, (x,y), 0, (255,0,0))

        common.qimshow([ [normalized(mode_count)],
                         [cv2.cvtColor(mode_filtered_copy, cv2.COLOR_Lab2BGR)],
                         [cv2.cvtColor(one_mode, cv2.COLOR_Lab2BGR)],
                         [cv2.cvtColor(imagelabcopy, cv2.COLOR_Lab2BGR)] ])
        for thresh in range(different_color_threshold, different_color_threshold + 1):
            absdiff = cv2.absdiff(imagelab, one_mode)
            b, g, r = cv2.split(absdiff)
            norm = cv2.sqrt(np.float32(b) * b + np.float32(g) * g + np.float32(r) * r)
            minval, maxval, _, _ = cv2.minMaxLoc(norm)
            print 'norm ranges from %f to %f' % (minval, maxval)
            retval, norm_limited = cv2.threshold(norm, thresh * 2, thresh * 2, cv2.THRESH_TRUNC)
            minval, maxval, _, _ = cv2.minMaxLoc(norm_limited)
            print 'norm_limited ranges from %f to %f' % (minval, maxval)

            deviatingpixels = (norm > thresh).astype(np.float32)
            deviatingpixelsrgb = np.uint8(cv2.cvtColor(deviatingpixels, cv2.COLOR_GRAY2BGR) * 255)
            cv2.circle(deviatingpixelsrgb, (x,y), common.NOTE_SIZE / 2, (255,0,255))


            circle_indices = np.where(circle != 0)
            x_indices = circle_indices[1] + x - common.NOTE_SIZE / 2
            y_indices = circle_indices[0] + y - common.NOTE_SIZE / 2
            onlycircledeviation = np.ones(deviatingpixels.shape, deviatingpixels.dtype)
            onlycircledeviation[(y_indices, x_indices)] = deviatingpixels[(y_indices, x_indices)]

            inverted = np.zeros(deviatingpixels.shape, np.uint8)
            inverted[onlycircledeviation == 0] = 255
            print 'number of matching pixels is %d, mode_count for this pixel is %d' % (cv2.countNonZero(inverted), int(mode_count[y][x]))

            common.qimshow([ [normalized(norm_limited)],
                             [normalized(onlycircledeviation)],
                             [deviatingpixelsrgb] ])
    '''


    kernel = np.ones((common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), np.uint8)
    similar_dilated = cv2.dilate(mode_count, kernel)
    local_max_at_zero = mode_count - similar_dilated
    local_max_mask = cv2.inRange(local_max_at_zero, 0.0, 1.0)

    max_similarity_mask = cv2.bitwise_and(local_max_mask, significant_color_mask)

    temp = normalized(mode_count)
    temp[max_similarity_mask == 0] = 0

    _, contours, _ = cv2.findContours(max_similarity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotatedimage = image.copy()
    for contour in contours:
        center, _ = cv2.minEnclosingCircle(contour)
        count = mode_count[center[1]][center[0]]
        if count > (2 * circleArea / 3):
            cv2.circle(annotatedimage, tuple(int(f) for f in center), int((count / 2000) * common.NOTE_SIZE / 2), (255,0,255))

    common.qimshow([ [image, image],
                     [cv2.cvtColor(mode_filtered, cv2.COLOR_Lab2BGR), normalized(mode_count)],
                     [max_similarity_mask, temp ],
                     [annotatedimage] ])


    #values, counts = np.unique(a[0], return_counts=True)
    #maxval = values[np.argmax(counts)]









    average = determine_average_colors(image, circle)

    hsv = cv2.cvtColor( median, cv2.COLOR_BGR2HSV )
    _, saturation, _ = cv2.split(hsv)
    no_color = cv2.inRange(saturation, 0, 24)
    color = cv2.bitwise_not(no_color)


    padded = np.pad(image, ((common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), (common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), (0,0)), 'constant')
    """
    grayimage = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    common.qimshow(grayimage)
    shifteds = []
    for y in range(circle.shape[0]):
        for x in range(circle.shape[1]):
            if circle[y][x] != 0:
                shifted = grayimage[y:y + image.shape[0], x:x + image.shape[1]]
                shifteds.append(np.float32(shifted))

    _, _, _, mean_corrected_norm = calculate_parameters_from_mat(shifteds)
    minval, maxval, _, _ = cv2.minMaxLoc(mean_corrected_norm)
    norm = np.zeros(mean_corrected_norm.shape, dtype=np.float32)
    norm = cv2.normalize(mean_corrected_norm, norm, 0, 1, cv2.NORM_MINMAX)
    common.qimshow(norm)
    """

    halfnotesize = common.NOTE_SIZE / 2


    paddedlab = cv2.cvtColor(padded, cv2.COLOR_BGR2Lab)
    common.qimshow(paddedlab)

    shifteds = []
    for y in range(circle.shape[0]):
        for x in range(circle.shape[1]):
            if circle[y][x] != 0:
                shifted = paddedlab[y:y + image.shape[0], x:x + image.shape[1]]
                shifteds.append(shifted)

    deviation = np.zeros(image.shape[:2], dtype=np.float32)
    for shifted in shifteds:
        absdiff = cv2.absdiff(shifted, medianlab)
        b, g, r = cv2.split(absdiff)
        norm = cv2.sqrt(np.float32(b) * b + g * g + r * r)
        deviatingpixels = (norm > different_color_threshold).astype(np.float32)
        #common.qimshow([median, shifted, absdiff, r, g, b, np.uint8(norm), deviatingpixels])
        deviation += deviatingpixels

    minval, maxval, _, _ = cv2.minMaxLoc(deviation, color)
    print minval, maxval

    masked_deviation = deviation.copy()
    masked_deviation[no_color == 255] = maxval

    minimal_deviation = cv2.inRange(masked_deviation, 0, 150)





    kernel = np.ones((common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), np.uint8)
    deviation_eroded = cv2.erode(deviation, kernel)

    local_min_at_zero = deviation - deviation_eroded
    local_min_only = cv2.inRange(local_min_at_zero, 0, 0)
    local_min_only[no_color == 255] = 0

    _, contours, _ = cv2.findContours(local_min_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotatedimage = image.copy()
    for contour in contours:
        center, _ = cv2.minEnclosingCircle(contour)
        cv2.circle(annotatedimage, tuple(int(f) for f in center), int(deviation[center[1]][center[0]] / 5), (0,0,255))
        print 'contour is at %s with deviation %s' % (center, deviation[center[1]][center[0]])

    common.qimshow([ [median],
                     [normalized(masked_deviation)],
                     [normalized(local_min_at_zero), local_min_only],
                     [annotatedimage]
                   ])

    return []

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    #image = cv2.imread('board.png')
    image = webcam.grab()

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))
    image, _ = common.correct_perspective(common.remove_color_cast(image, calibrationdata), calibrationdata, False)

    #notes = findnotes(image[0:200,650:900])
    #notes = findnotes(image[600:,0:100])
    #notes = findnotes(image[550:700,350:500])
    #notes = findnotes(image[:,350:550])
    notes = findnotes(image)

