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

def findnotes(image):
    #averages = determine_average_colors(image)

    common.qimshow(image)

    median = cv2.medianBlur(image, 11)
    common.qimshow(median)

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

    deviation = np.zeros(image.shape[:2], dtype=np.float32)
    for y in range(circle.shape[0]):
        for x in range(circle.shape[1]):
            if circle[y][x] != 0:
                shifted = padded[y:y + image.shape[0], x:x + image.shape[1]]
                diff = cv2.absdiff(shifted, median)
                squaredsum = np.sum(diff, axis=2)
                deviation += squaredsum / 3


    deviation_norm = np.zeros(deviation.shape, dtype=np.float32)
    deviation_norm = cv2.normalize(deviation, deviation_norm, 0, 1, cv2.NORM_MINMAX)
    deviation_norm = 1.0 - deviation_norm


    kernel = np.ones((common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), np.uint8)
    deviation_dil = cv2.dilate(deviation_norm, kernel)
    deviation_sub = deviation_norm - deviation_dil
    minval, maxval, _, _ = cv2.minMaxLoc(deviation_sub)
    print 'min was %f, max was %f' % (minval, maxval)

    deviation_subclip = (deviation_sub > -0.0001).astype(np.float32)

    hsv = cv2.cvtColor( median, cv2.COLOR_BGR2HSV )
    _, saturation, _ = cv2.split(hsv)
    color_only = cv2.inRange(saturation, 25, 255)

    deviation_subclip2 = cv2.inRange(deviation_subclip, 0.5, 1.0)
    deviation_subclip2[color_only == 0] = 0

    _, contours, _ = cv2.findContours(deviation_subclip2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotatedimage = image.copy()
    for contour in contours:
        center, _ = cv2.minEnclosingCircle(contour)
        cv2.circle(annotatedimage, tuple(int(f) for f in center), common.NOTE_SIZE / 2, (0,0,255))
        print 'contour is at %s' % (center,)

    common.qimshow([ [deviation_norm],
                     [deviation_dil],
                     [deviation_sub],
                     [deviation_subclip],
                     [color_only],
                     [deviation_subclip2],
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

