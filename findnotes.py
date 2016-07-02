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
    common.qimshow(averages)
    return averages

def findnotes(image):
    #averages = determine_average_colors(image)

    common.qimshow(image)

    median = cv2.medianBlur(image, common.NOTE_SIZE - 1)
    common.qimshow(median)

    padded = np.pad(image, ((common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), (common.NOTE_SIZE / 2, common.NOTE_SIZE / 2), (0,0)), 'constant')
    common.qimshow(padded)

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

    #deviation_uint = np.uint8(deviation_norm * 255)
    #peaks1 = cv2.adaptiveThreshold(deviation_uint, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, common.NOTE_SIZE / 2, -3)
    #peaks2 = cv2.adaptiveThreshold(deviation_uint, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, common.NOTE_SIZE / 2, -5)
    #peaks3 = cv2.adaptiveThreshold(deviation_uint, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, common.NOTE_SIZE / 2, -7)
    #common.qimshow([deviation_uint, peaks1, peaks2, peaks3])

    kernel = np.ones((common.NOTE_SIZE - 9, common.NOTE_SIZE - 9), np.uint8)
    deviation_dil = cv2.dilate(deviation_norm, kernel)
    deviation_sub = deviation_norm - deviation_dil
    minval, maxval, _, _ = cv2.minMaxLoc(deviation_sub)
    print 'min was %f, max was %f' % (minval, maxval)

    norm = np.zeros(deviation_dil.shape, dtype=np.float32)
    norm = cv2.normalize(deviation_dil, norm, 0, 1, cv2.NORM_MINMAX)
    common.qimshow(norm)

    deviation_sub2 = (deviation_norm < deviation_dil).astype(np.float32)

    common.qimshow([deviation_norm, deviation_dil, deviation_sub, deviation_sub2])
    peaks = (deviation_sub == 0).astype(np.float32)


    return []

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    image = cv2.imread('board1.png')

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))
    image, _ = common.correct_perspective(image, calibrationdata, False)

    notes = findnotes(image[:200, 100:-100])

