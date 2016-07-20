#!/usr/bin/env python2

# Copyright 2016 Maurice van der Pot <griffon26@kfk4ever.com>
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

import imagefuncs
import webcam

from PyQt5 import QtWidgets

# To make opencv2 compatible with the opencv3 API we use
if cv2.__version__.startswith('2'):
    cv2.COLOR_BGR2Lab = cv2.COLOR_BGR2LAB

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
    diameter = imagefuncs.NOTE_SIZE
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

    different_color_threshold = 31

    circle = createCircularKernel()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)
    retval, color = cv2.threshold(saturation, 25, 1, cv2.THRESH_BINARY)

    sum_of_color_in_circle = cv2.filter2D(color, cv2.CV_32F, circle)

    circleArea = cv2.countNonZero(circle)
    retval, significant_color_mask = cv2.threshold(sum_of_color_in_circle, 2 * circleArea / 3, 255, cv2.THRESH_BINARY)
    significant_color_mask = np.uint8(significant_color_mask)



    # 3-channel mode
    padsize = imagefuncs.NOTE_SIZE / 2

    color_absence = np.float32((1 - color) * different_color_threshold)

    padded = np.pad(image, ((padsize, padsize), (padsize, padsize), (0,0)), 'constant')
    paddedlab = cv2.cvtColor(padded, cv2.COLOR_BGR2Lab)
    color_absence_padded = np.pad(color_absence, ((padsize, padsize), (padsize, padsize)), 'constant', constant_values = different_color_threshold)

    h, w, c = padded.shape
    paddedmerged = np.zeros((h, w, c + 1), dtype=np.uint8)
    paddedmerged[:,:,:3] = paddedlab
    paddedmerged[:,:,3] = color_absence_padded
    paddedpacked = paddedmerged.view(np.uint32)

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

        count = np.sum(counts[eucl_diffs <= different_color_threshold])

        mode_count[y_offset][x_offset] = count

    #imagefuncs.qimshow(normalized(mode_count))

    kernel = np.ones((imagefuncs.NOTE_SIZE / 2, imagefuncs.NOTE_SIZE / 2), np.uint8)
    similar_dilated = cv2.dilate(mode_count, kernel)
    local_max_at_zero = mode_count - similar_dilated
    local_max_mask = cv2.inRange(local_max_at_zero, 0.0, 1.0)

    max_similarity_mask = cv2.bitwise_and(local_max_mask, significant_color_mask)

    temp = normalized(mode_count)
    temp[max_similarity_mask == 0] = 0

    retvals = cv2.findContours(max_similarity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cv2.__version__.startswith('3'):
        _, contours, _ = retvals
    else:
        contours, _ = retvals

    annotatedimage = image.copy()
    positions = []
    for contour in contours:
        center, _ = cv2.minEnclosingCircle(contour)
        intcenter = tuple(int(f) for f in center)
        count = mode_count[center[1]][center[0]]
        if count > (2 * circleArea / 3):
            cv2.circle(annotatedimage, intcenter, int((count / 2000) * imagefuncs.NOTE_SIZE / 2), (255,0,255))
            cv2.circle(annotatedimage, intcenter, imagefuncs.NOTE_SIZE / 2, (255,0,0))
            positions.append(intcenter)

    #imagefuncs.qimshow(annotatedimage)

    return positions

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    image = webcam.grab()

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))
    image, _ = imagefuncs.correct_perspective(imagefuncs.remove_color_cast(image, calibrationdata), calibrationdata, False)

    notes = findnotes(image)

