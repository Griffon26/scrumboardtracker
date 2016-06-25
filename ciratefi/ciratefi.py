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

    if abs(contrast_correction_factor) <= thresh_contrast or \
       abs(contrast_correction_factor) >= 1.0 / thresh_contrast or \
       abs(brightness_correction_factor) > thresh_brightness:
        correlation = 0.0
    else:
        correlation = correlation_coef

    return correlation

def remove_gradient(note):
    means = cv2.mean(note)
    background = np.zeros(note.shape, dtype=note.dtype)
    background[:] = (int(means[0]), int(means[1]), int(means[2]))
    gradient = cv2.GaussianBlur(note, (notesize - 1, notesize - 1), 0)

    note_without_gradient = note - gradient + background

    qimshow([ [note, gradient], [note_without_gradient, background] ])

    return note_without_gradient

def flatten_disc(img, center, radius):
    values = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            diff_x = x - center[0]
            diff_y = y - center[1]

            if diff_x * diff_x + diff_y * diff_y < radius * radius:
                values.append(img[y][x])

    return np.array(values)

if __name__ == "__main__":

    app = QApplication(sys.argv)

    note = cv2.imread('img3note.png', 1)
    board = cv2.imread('../photos/downscaled/img2.jpg', 1)
    notesize = min(note.shape[:2])

    note_cropped = submatrix(note, note.shape[1] / 2, note.shape[0] / 2, notesize)

    note_cropped = remove_gradient(note_cropped)

    note = cv2.cvtColor(note_cropped, cv2.COLOR_BGR2GRAY)
    board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

    nr_of_radii = 13
    min_nr_of_first_grade_candidates = 10
    percentage_of_first_grade_candidates = 0.1
    min_nr_of_second_grade_candidates = 10
    percentage_of_second_grade_candidates = 1
    thresh_contrast = 0.1
    thresh_brightness = 255.0
    threshold3 = 0.5


    print 'Performing cifi'

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


    candidates_with_correlation = []
    board_with_markers = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

    for x in range(board.shape[1]):
        for y in range(board.shape[0]):
            C_A = np.array([ cifi_board[k][y][x] for k in range(nr_of_radii) ])
            C_Q = np.array(cifi_means)


            correlation = calculate_correlation(C_Q, C_A, thresh_contrast, thresh_brightness)
            candidates_with_correlation.append( (correlation, x, y) )

    nr_of_first_grade_candidates = int((len(candidates_with_correlation) * percentage_of_first_grade_candidates) / 100)
    if nr_of_first_grade_candidates < min_nr_of_first_grade_candidates:
        nr_of_first_grade_candidates = min(len(candidates_with_correlation), min_nr_of_first_grade_candidates)
    first_grade_candidates = sorted([ (x, y) for _, x, y in sorted(candidates_with_correlation, reverse=True)[:nr_of_first_grade_candidates] ])

    print 'first_grade_candidates: %s' % first_grade_candidates

    for x, y in first_grade_candidates:
        cv2.circle(board_with_markers, (x, y), 1, (255, 0, 255))

    qimshow([ [note, board_with_markers],
              cifi_masks,
              cifi_fmasks ]
              )

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

    candidates_with_correlation = []
    for i, this_candidate_means in enumerate(candidate_means):
        x, y = first_grade_candidates[i]

        this_candidate_rascorrs = []

        max_rascorr = -2
        max_cshift = None

        for cshift in range(nr_of_rotation_angles):
            rotated_rafi_means = np.roll(rafi_means, cshift)

            rascorr = calculate_correlation(np.array(this_candidate_means), rotated_rafi_means, thresh_contrast, thresh_brightness)

            this_candidate_rascorrs.append(rascorr)

            if rascorr > max_rascorr:
                max_rascorr = rascorr
                max_cshift = cshift

        candidates_with_correlation.append( (max_rascorr, x, y, max_cshift) )

    nr_of_second_grade_candidates = int((len(candidates_with_correlation) * percentage_of_second_grade_candidates) / 100)
    if nr_of_second_grade_candidates < min_nr_of_second_grade_candidates:
        nr_of_second_grade_candidates = min(len(candidates_with_correlation), min_nr_of_second_grade_candidates)
    second_grade_candidates = sorted([ (x, y, cshift) for _, x, y, cshift in
                                       sorted(candidates_with_correlation, reverse=True)[:nr_of_second_grade_candidates] ])

    for x, y, cshift in second_grade_candidates:
        print 'best rotation for second grade candidate at %d,%d is %d' % (x, y, cshift * 10)
        cv2.circle(board_with_markers, (x, y), 3, (255, 0, 0))
        cv2.circle(board_with_markers, (x, y), 4, (255, 0, 0))

    qimshow([board_with_markers])

    print 'Performing tefi'

    rotated_templates = []
    for cshift in range(nr_of_rotation_angles):
        rotation_matrix = cv2.getRotationMatrix2D((notesize / 2, notesize / 2), -cshift * 10, 1)
        rotated_note = cv2.warpAffine(note, rotation_matrix, (notesize, notesize))
        masked_rotated_note = flatten_disc(rotated_note, (notesize / 2, notesize / 2), notesize / 2)
        rotated_templates.append(masked_rotated_note)

    candidates_with_correlation = []
    for x, y, cshift in second_grade_candidates:
        for x2 in [x - 1, x, x + 1]:
            for y2 in [y - 1, y, y + 1]:
                possible_match = submatrix(board, x2, y2, notesize)
                masked_possible_match = flatten_disc(possible_match, (notesize / 2, notesize / 2), notesize / 2)
                corr = calculate_correlation(rotated_templates[cshift], masked_possible_match, thresh_contrast, thresh_brightness)
                #print 'correlation with second grade candidate at (%d,%d) is %f' % (x2, y2, corr)
                candidates_with_correlation.append( (corr, x2, y2) )

    candidates_with_correlation.sort(reverse = True)

    final_match = candidates_with_correlation[0]

    if final_match[0] < threshold3:
        print 'no match was found. The best was at (%d,%d) with correlation %f' % (final_match[1], final_match[2], final_match[0])
        qimshow(submatrix(board, final_match[1], final_match[2], notesize))
    else:
        print 'final match is at (%d,%d) with correlation %f' % (final_match[1], final_match[2], final_match[0])
        qimshow(submatrix(board, final_match[1], final_match[2], notesize))
        for corr, x, y in candidates_with_correlation:
            diff_x = x - final_match[1]
            diff_y = y - final_match[2]
            if diff_x * diff_x + diff_y * diff_y > notesize * notesize:
                print 'second best match is at (%d,%d) with correlation %f' % (x, y, corr)
                qimshow(submatrix(board, x, y, notesize))
                break




