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

def findnotes(image):
    #averages = determine_average_colors(image)

    #common.qimshow(image)

    median = cv2.medianBlur(image, 11)
    #common.qimshow(median)
    medianlab = cv2.cvtColor(median, cv2.COLOR_BGR2Lab)

    different_color_threshold = 21

    common.qimshow(image)

    """
    for x, y in [ (441, 756), (1220, 587), (580, 644), (467, 756), (44, 750) ]:

        onemedian = median.copy()
        onemedian[:] = median[y][x]

        imagelab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        onemedianlab = cv2.cvtColor(onemedian, cv2.COLOR_BGR2Lab)

        common.qimshow(onemedian)
        for thresh in range(different_color_threshold, different_color_threshold + 1):
            absdiff = cv2.absdiff(imagelab, onemedianlab)
            b, g, r = cv2.split(absdiff)
            norm = cv2.sqrt(np.float32(b) * b + g * g + r * r)
            deviatingpixels = (norm > thresh).astype(np.float32)
            common.qimshow(normalized(deviatingpixels))
    """


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

    common.qimshow([ ['saturation', saturation],
                     #[sum_of_color_in_circle, normalized(sum_of_color_in_circle)],
                     #[color_dilated, normalized(color_dilated)],
                     #[local_max_at_zero, normalized(local_max_at_zero)],
                     ['local max', local_max_mask, local_max_image],
                     ['significant color', significant_color_mask, significant_color_image],
                     ['max color', max_color_mask, max_color_image],
                     ['found notes', annotatedimage] ])







    padsize = common.NOTE_SIZE / 2

    padded = np.pad(image, ((padsize, padsize), (padsize, padsize), (0,0)), 'constant')
    paddedlab = cv2.cvtColor(padded, cv2.COLOR_BGR2Lab)
    b, g, r = cv2.split(paddedlab)
    paddedmerged = np.uint32(b) * 256 * 256 + g * 256 + r


    circle_indices = np.where(circle != 0)

    mode_filtered = image.copy()

    mode_count = image.astype(np.uint16)
    color_indices = np.where(significant_color_mask != 0)
    for y_offset, x_offset in zip(*color_indices):
        y_indices = circle_indices[0] + y_offset
        x_indices = circle_indices[1] + x_offset

        colors = paddedmerged[(y_indices, x_indices)]
        values, counts = np.unique(colors, return_counts=True)
        mode_packed = values[np.argmax(counts)]
        mode = np.array([(mode_packed >> 16) & 255,
                         (mode_packed >>  8) & 255,
                         (mode_packed      ) & 255])
        mode_filtered[y_offset][x_offset] = mode

        lab_values = np.stack( (np.bitwise_and(np.right_shift(values, 16), 255),
                                np.bitwise_and(np.right_shift(values, 8), 255),
                                np.bitwise_and(values, 255)), axis = -1 )
        absdiff = np.absolute(lab_values - mode)
        eucl_diffs = np.sqrt(np.sum(absdiff * absdiff, axis=1))
        count = len(eucl_diffs[eucl_diffs < 25])

        mode_count[y_offset][x_offset] = count




    kernel = np.ones((common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), np.uint8)
    similar_dilated = cv2.dilate(mode_count, kernel)
    local_max_at_zero = np.float32(mode_count) - similar_dilated
    local_max_mask = cv2.inRange(local_max_at_zero, 0.0, 1.0)

    max_similarity_mask = cv2.bitwise_and(local_max_mask, significant_color_mask)

    temp = normalized(np.float32(mode_count))
    temp[max_similarity_mask == 0] = 0

    _, contours, _ = cv2.findContours(max_similarity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotatedimage = image.copy()
    for contour in contours:
        center, _ = cv2.minEnclosingCircle(contour)
        cv2.circle(annotatedimage, tuple(int(f) for f in center), common.NOTE_SIZE / 2, (255,0,255))

    common.qimshow([ [image, image],
                     [mode_filtered, normalized(np.float32(mode_count))],
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
    notes = findnotes(image)

