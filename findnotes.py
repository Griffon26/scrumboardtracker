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

def remove_color_cast(image):
    bgr_planes = cv2.split(image)

    backgroundpos = calibrationdata['background']
    average_bgr = image[backgroundpos[1]][backgroundpos[0]]
    print 'bgr of scrumboard background:', average_bgr

    for i in xrange(3):
        thresh = average_bgr[i]
        plane = bgr_planes[i]
        print "Showing plane"
        #qimshow(plane);

        retval, mask = cv2.threshold(plane, thresh, 255, cv2.THRESH_BINARY);
        print "Showing mask"
        #qimshow(mask)

        highvalues = (plane - thresh) * (128.0 / (255 - thresh)) + 128;
        highvalues = highvalues.astype(np.uint8)

        highvalues_masked = cv2.bitwise_and(highvalues, mask)
        print "Showing scaled high values"
        #qimshow(highvalues_masked)

        mask = 255 - mask;
        lowvalues = cv2.bitwise_and(plane, mask)
        print "Showing low values"
        #qimshow(lowvalues)

        lowvalues = lowvalues * 128.0 / thresh
        lowvalues = lowvalues.astype(np.uint8)
        print "Showing scaled low values"
        #qimshow(lowvalues)

        bgr_planes[i] = lowvalues + highvalues_masked
        print "Showing scaled plane"
        #qimshow(bgr_planes[i])

    correctedimage = cv2.merge(bgr_planes)
    correctedimage = correctedimage.astype(np.uint8)
    print "Showing corrected image"
    #qimshow(correctedimage)

    return correctedimage

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
    for x, y in [ (1220, 587), (580, 644), (467, 756), (44, 750) ]:

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
         
    padded = np.pad(image, ((common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), (common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), (0,0)), 'constant')

    circle = createCircularKernel()

    average = determine_average_colors(image, circle)


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

    shifteds = []
    for y in range(circle.shape[0]):
        for x in range(circle.shape[1]):
            if circle[y][x] != 0:
                shifted = padded[y:y + image.shape[0], x:x + image.shape[1]]
                shiftedlab = cv2.cvtColor(shifted, cv2.COLOR_BGR2Lab)
                shifteds.append(shiftedlab)

    deviation = np.zeros(image.shape[:2], dtype=np.float32)
    for shifted in shifteds:
        absdiff = cv2.absdiff(shifted, medianlab)
        b, g, r = cv2.split(absdiff)
        norm = cv2.sqrt(np.float32(b) * b + g * g + r * r)
        deviatingpixels = (norm > different_color_threshold).astype(np.float32)
        #common.qimshow([median, shifted, absdiff, r, g, b, np.uint8(norm), deviatingpixels])
        deviation += deviatingpixels

    hsv = cv2.cvtColor( median, cv2.COLOR_BGR2HSV )
    _, saturation, _ = cv2.split(hsv)
    no_color = cv2.inRange(saturation, 0, 24)
    color = cv2.bitwise_not(no_color)

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

    _, contours, _ = cv2.findContours(minimal_deviation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotatedimage = image.copy()
    for contour in contours:
        center, _ = cv2.minEnclosingCircle(contour)
        cv2.circle(annotatedimage, tuple(int(f) for f in center), common.NOTE_SIZE / 2, (0,0,255))
        print 'contour is at %s' % (center,)

    common.qimshow([ [median],
                     [normalized(deviation), no_color, normalized(masked_deviation), minimal_deviation],
                     [annotatedimage]
                   ])

    return []

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    #image = cv2.imread('board.png')
    image = webcam.grab()

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))
    image, _ = common.correct_perspective(remove_color_cast(image), calibrationdata, False)

    #notes = findnotes(image[600:800,200:500])
    notes = findnotes(image)

