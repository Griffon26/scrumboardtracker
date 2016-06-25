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

sys.path.append('..')

import common

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class ImageLabel(QLabel):
    def __init__(self, image):
        QLabel.__init__(self, None)

        self.image = image
        self.redraw()

    def redraw(self):
        pixmap = common.cvimage_to_qpixmap(self.image)
	self.setGeometry(300, 300, pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)

class ImageDialog(QDialog):
    
    def __init__(self, images, text=None):
        super(ImageDialog, self).__init__()

        if not isinstance(images, (list, tuple)):
            images = [images]

        if not isinstance(images[0], (list, tuple)):
            images = [images]

        vbox = QVBoxLayout()

        if text != None:
            vbox.addWidget(QLabel(text))

        for imagelist in images:
            hbox = QHBoxLayout()
            for image in imagelist:
                hbox.addWidget(ImageLabel(image))
            vbox.addLayout(hbox)

        buttonBox = QDialogButtonBox(self)
        buttonBox.setGeometry(QRect(150, 250, 341, 32))
        buttonBox.setOrientation(Qt.Horizontal)
        buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        buttonBox.setObjectName("buttonBox")
        vbox.addWidget(buttonBox)

        self.setLayout(vbox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

def qimshow(images, text=None):
    dlg = ImageDialog(images, text)
    if dlg.exec_() != 1:
        raise Exception('Aborting. User pressed cancel.')

# Determines the start and end of source and destination ranges for copying
# 'dest_size' pixels out of a range of 'source_size' pixels starting at offset
# 'offset'. The ranges given by this function can be used to copy a submatrix
# from a larger matrix without reading outside the source matrix or writing
# outside the destination matrix.
def determine_copy_ranges(offset, source_size, dest_size):

    dest_start = 0
    dest_end = dest_size

    src_start = offset
    src_end = offset + dest_size

    if src_start < 0:
        dest_start += -src_start
        src_start = 0
    if src_end > source_size:
        dest_end -= src_end - source_size
        src_end = source_size

    return src_start, src_end, dest_start, dest_end

def submatrix(bitmap_in, x, y, size):

    shape = [size, size]
    if len(bitmap_in.shape) == 3:
        shape.append(bitmap_in.shape[2])
    bitmap_out = np.zeros(shape, dtype=bitmap_in.dtype)

    src_row_start, src_row_end, dest_row_start, dest_row_end = determine_copy_ranges(y - size / 2, bitmap_in.shape[0], size)
    src_col_start, src_col_end, dest_col_start, dest_col_end = determine_copy_ranges(x - size / 2, bitmap_in.shape[1], size)

    bitmap_out[dest_row_start:dest_row_end,
               dest_col_start:dest_col_end] = bitmap_in[src_row_start:src_row_end,
                                                        src_col_start:src_col_end]
    return bitmap_out

def tilde(x):
    return x - cv2.mean(x)[0]

def calculate_correlation(x, y, thresh_contrast, thresh_brightness):
    x = x.flatten()
    y = y.flatten()

    mean_corrected_x = tilde(x)
    mean_corrected_y = tilde(y)

    mean_corrected_xy = np.dot(mean_corrected_x, mean_corrected_y)
    mean_corrected_x_squared = np.dot(mean_corrected_x, mean_corrected_x)
    contrast_correction_factor = mean_corrected_xy / mean_corrected_x_squared

    brightness_correction_factor = cv2.mean(y)[0] - contrast_correction_factor * cv2.mean(x)[0]

    correlation_coef = (contrast_correction_factor * mean_corrected_x_squared) / (cv2.norm(mean_corrected_x) * cv2.norm(mean_corrected_y))

    #print 'contrast_correction_factor', contrast_correction_factor
    #print 'brightness_correction_factor', brightness_correction_factor

    if abs(contrast_correction_factor) <= thresh_contrast or \
       abs(contrast_correction_factor) >= 1.0 / thresh_contrast or \
       abs(brightness_correction_factor) > thresh_brightness:
        correlation = 0.0
    else:
        correlation = correlation_coef

    return correlation

if __name__ == "__main__":

    app = QApplication(sys.argv)

    note = cv2.imread('note.png', 1)
    board = cv2.imread('img5.jpg', 1)
    notesize = min(note.shape[:2])

    note_cropped = submatrix(note, note.shape[1] / 2, note.shape[0] / 2, notesize)
    #qimshow([note, note_cropped])

    note = cv2.cvtColor(note_cropped, cv2.COLOR_BGR2GRAY)
    board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

    nr_of_radii = 13

    cifi_masks = []
    cifi_means = []
    cifi_fmasks = []
    cifi_filtered = []
    cifi_board = []
    for i in range(0, nr_of_radii):
        mask = np.zeros((notesize, notesize), dtype=np.uint8)
        cv2.circle(mask, (notesize / 2, notesize / 2), int((i * (notesize / 2.0)) / nr_of_radii), (255,255,255))
        cifi_masks.append(mask)
        cifi_means.append(cv2.mean(note, mask)[0])

        maskcount = cv2.countNonZero(mask)

        fmask = np.float32(mask) / 255
        fmask = fmask / maskcount
        cifi_fmasks.append(fmask)

        cifi_board.append(cv2.filter2D(board, -1, fmask))

    threshold1 = 0.97
    threshold2 = 0.6
    thresh_contrast = 0.1
    thresh_brightness = 255.0


    print 'Performing cifi'
    matches = []
    first_grade_candidates = []
    ciscorr = np.zeros(board.shape, dtype=np.float32)
    board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    board_with_markers = board.copy()

    for x in range(board.shape[1]):
        for y in range(board.shape[0]):
            C_A = np.array([ cifi_board[k][y][x] for k in range(nr_of_radii) ])
            C_Q = np.array(cifi_means)


            ciscorr[y][x] = calculate_correlation(C_Q, C_A, thresh_contrast, thresh_brightness)
            if ciscorr[y][x] > threshold1:
                print '%s matches %s with probability %s' % (np.uint8(C_Q), C_A, ciscorr[y][x])
                matches.extend([ C_Q, C_A, [0] * nr_of_radii ])
                cv2.circle(board_with_markers, (x, y), 1, (255, 0, 255))
                first_grade_candidates.append( (x, y) )

    qimshow([ [note, ciscorr, board_with_markers],
              cifi_masks,
              cifi_fmasks ]
              )

    #first_grade_candidates = [(408, 234), (409, 239), (412, 246)]
    print 'first_grade_candidates: %s' % first_grade_candidates

    print 'Performing rafi'

    nr_of_rotation_angles = 36

    rafi_masks = []
    rafi_means = []
    for i in range(nr_of_rotation_angles):
        angle = -(math.pi / 2) + (i * 2 * math.pi) / 36

        halfnotesize = notesize / 2

        p1 = (halfnotesize, halfnotesize)
        p2 = (int(halfnotesize + halfnotesize * math.cos(angle)),
              int(halfnotesize + halfnotesize * math.sin(angle)))

        mask = np.zeros((notesize, notesize), dtype=np.uint8)
        cv2.line(mask, p1, p2, (255,255,255))
        rafi_masks.append(mask)
        rafi_means.append(cv2.mean(note, mask)[0])

    rafi_means = np.array(rafi_means)
    print 'rafi_means', [int(x) for x in rafi_means]

    candidate_means = []
    for x, y in first_grade_candidates:
        candidate = submatrix(board, x, y, notesize)
        #qimshow(candidate)
        this_candidate_means = []
        for mask in rafi_masks:
            this_candidate_means.append(cv2.mean(candidate, mask)[0])
        #print 'means for candidate at (%d, %d) are %s' % (x, y, [int(x) for x in this_candidate_means])
        candidate_means.append(this_candidate_means)

    second_grade_candidates = []
    for i, this_candidate_means in enumerate(candidate_means):
        this_candidate_rascorrs = []

        max_rascorr = -2
        max_cshift = None

        for cshift in range(nr_of_rotation_angles):
            rotated_rafi_means = np.roll(rafi_means, cshift)

            #print 'this means:', this_candidate_means
            #print 'rotatedraf:', rotated_rafi_means
            rascorr = calculate_correlation(np.array(this_candidate_means), rotated_rafi_means, thresh_contrast, thresh_brightness)
            #print 'rascorr for cshift %d is %f' % (cshift, rascorr)

            this_candidate_rascorrs.append(rascorr)

            if rascorr > max_rascorr:
                max_rascorr = rascorr
                max_cshift = cshift

        #print 'best rotation for candidate %d at %d,%d is %d with rascorr %f' % (i, first_grade_candidates[i][0],
        #                                                            first_grade_candidates[i][1], max_cshift * 10, max_rascorr)
        if max_rascorr >= threshold2:
            print 'upgrading candidate at %d,%d with rascorr %f and best rotation %d to second grade candidate' % \
                (first_grade_candidates[i][0], first_grade_candidates[i][1], max_rascorr, max_cshift * (360 / nr_of_rotation_angles))
            second_grade_candidates.append(first_grade_candidates[i])
            cv2.circle(board_with_markers, first_grade_candidates[i], 1, (255, 0, 0))

    qimshow([board_with_markers])


