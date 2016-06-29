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

def calculate_parameters_from_list(list_of_x):
    mean_x = np.mean(list_of_x, dtype=np.float32)

    mean_corrected_x = list_of_x - mean_x

    mean_corrected_x_squared = np.dot(mean_corrected_x, mean_corrected_x)

    mean_corrected_x_norm = math.sqrt(mean_corrected_x_squared)

    return mean_x, mean_corrected_x, mean_corrected_x_squared, mean_corrected_x_norm

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

def calculate_image_correlation(stack_of_x, stack_of_y, thresh_contrast, thresh_brightness):

    mean_x, mean_corrected_x, mean_corrected_x_squared, mean_corrected_x_norm = calculate_parameters_from_list(stack_of_x)
    mean_y, mean_corrected_y, mean_corrected_y_squared, mean_corrected_y_norm = calculate_parameters_from_mat(stack_of_y)

    element_wise_xy = [np.float32(y) * x for x, y in zip(mean_corrected_x, mean_corrected_y)]
    mean_corrected_xy = np.zeros(mean_y.shape, dtype=np.float32)
    for xy in element_wise_xy:
        mean_corrected_xy = cv2.add(mean_corrected_xy, xy)

    contrast_correction_factor = mean_corrected_xy / mean_corrected_x_squared

    brightness_correction_factor = mean_y - contrast_correction_factor * mean_x

    correlation_coef = (contrast_correction_factor * mean_corrected_x_squared) / (mean_corrected_x_norm * mean_corrected_y_norm)

    low_contrast_indices = abs(contrast_correction_factor) <= thresh_contrast
    high_contrast_indices = abs(contrast_correction_factor) >= (1.0 / thresh_contrast)
    high_brightness_indices = abs(brightness_correction_factor) > thresh_brightness

    correlation_coef[low_contrast_indices] = 0
    correlation_coef[high_contrast_indices] = 0
    correlation_coef[high_brightness_indices] = 0

    return correlation_coef


def remove_gradient(note):
    means = cv2.mean(note)
    background = np.zeros(note.shape, dtype=note.dtype)
    background[:] = (int(means[0]), int(means[1]), int(means[2]))
    gradient = cv2.GaussianBlur(note, (notesize - 1, notesize - 1), 0)

    note_without_gradient = note - gradient + background

    common.qimshow([ [note, gradient], [note_without_gradient, background] ])

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

class Ciratefi:

    def __init__(self, board, notesize, settings = {}, debug=False):
        self.settings = {
            'nr_of_radii' : 13,
            'nr_of_rotation_angles' : 36,
            'min_nr_of_first_grade_candidates' : 10,
            'percentage_of_first_grade_candidates' : 0.1,
            'min_nr_of_second_grade_candidates' : 10,
            'percentage_of_second_grade_candidates' : 1,
            'thresh_contrast' : 0.1,
            'thresh_brightness' : 255.0,
            'thresh_confidence' : 0.5
        }
        self.settings.update(settings)
        self.notesize = notesize
        self.board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        self.debug = debug

        print 'Performing cifi on the board'
        self.cifi_masks = []
        self.cifi_board = []
        for i in range(0, self.settings['nr_of_radii']):
            mask = np.zeros((notesize, notesize), dtype=np.uint8)
            cv2.circle(mask, (notesize / 2, notesize / 2), int((i * (notesize / 2.0)) / self.settings['nr_of_radii']), (255,255,255))
            self.cifi_masks.append(mask)

            maskcount = cv2.countNonZero(mask)

            fmask = np.float32(mask) / 255
            fmask = fmask / maskcount

            self.cifi_board.append(cv2.filter2D(self.board, -1, fmask))

    def _cifi(self, note):
        print 'Performing cifi'
        cifi_means = []
        for i in range(0, self.settings['nr_of_radii']):
            cifi_means.append(cv2.mean(note, self.cifi_masks[i])[0])

        candidates_with_correlation = []

        C_A = [ np.float32(self.cifi_board[k]) for k in range(self.settings['nr_of_radii']) ]
        C_Q = np.array(cifi_means)

        correlation = calculate_image_correlation(C_Q, C_A, self.settings['thresh_contrast'],
                                                            self.settings['thresh_brightness'])

        nr_of_candidates = cv2.countNonZero(correlation)
        nr_of_first_grade_candidates = int((nr_of_candidates * self.settings['percentage_of_first_grade_candidates']) / 100)
        if nr_of_first_grade_candidates < self.settings['min_nr_of_first_grade_candidates']:
            nr_of_first_grade_candidates = min(nr_of_candidates, self.settings['min_nr_of_first_grade_candidates'])

        sorted_indices = np.argsort(correlation, None)[-nr_of_first_grade_candidates:]
        sorted_indexpairs = np.unravel_index(sorted_indices, correlation.shape)

        first_grade_candidates = sorted([ (x, y) for y, x in zip(*sorted_indexpairs) ])

        return first_grade_candidates

    def _rafi(self, note, first_grade_candidates):
        print 'Performing rafi'

        rafi_masks = []
        rafi_means = []
        for i in range(self.settings['nr_of_rotation_angles']):
            angle = -(math.pi / 2) + (i * 2 * math.pi) / 36

            halfnotesize = self.notesize / 2

            p1 = (halfnotesize, halfnotesize)
            p2 = (int(halfnotesize + halfnotesize * math.cos(angle)),
                  int(halfnotesize + halfnotesize * math.sin(angle)))

            mask = np.zeros((self.notesize, self.notesize), dtype=np.uint8)
            cv2.line(mask, p1, p2, (255,255,255))
            rafi_masks.append(mask)
            rafi_means.append(cv2.mean(note, mask)[0])

        rafi_means = np.array(rafi_means)

        candidate_means = []
        for x, y in first_grade_candidates:
            candidate = common.submatrix(self.board, x, y, self.notesize)
            this_candidate_means = []
            for mask in rafi_masks:
                this_candidate_means.append(cv2.mean(candidate, mask)[0])
            candidate_means.append(this_candidate_means)

        candidates_with_correlation = []
        for i, this_candidate_means in enumerate(candidate_means):
            x, y = first_grade_candidates[i]

            this_candidate_rascorrs = []

            max_rascorr = -2
            max_cshift = None

            for cshift in range(self.settings['nr_of_rotation_angles']):
                rotated_rafi_means = np.roll(rafi_means, cshift)

                rascorr = calculate_correlation(np.array(this_candidate_means),
                                                rotated_rafi_means,
                                                self.settings['thresh_contrast'],
                                                self.settings['thresh_brightness'])

                this_candidate_rascorrs.append(rascorr)

                if rascorr > max_rascorr:
                    max_rascorr = rascorr
                    max_cshift = cshift

            candidates_with_correlation.append( (max_rascorr, x, y, max_cshift) )

        nr_of_second_grade_candidates = int((len(candidates_with_correlation) * self.settings['percentage_of_second_grade_candidates']) / 100)
        if nr_of_second_grade_candidates < self.settings['min_nr_of_second_grade_candidates']:
            nr_of_second_grade_candidates = min(len(candidates_with_correlation), self.settings['min_nr_of_second_grade_candidates'])
        second_grade_candidates = sorted([ (x, y, cshift) for _, x, y, cshift in
                                           sorted(candidates_with_correlation, reverse=True)[:nr_of_second_grade_candidates] ])

        return second_grade_candidates

    def _tefi(self, note, second_grade_candidates):
        print 'Performing tefi'

        rotated_templates = []
        for cshift in range(self.settings['nr_of_rotation_angles']):
            rotation_matrix = cv2.getRotationMatrix2D((self.notesize / 2, self.notesize / 2), -cshift * 10, 1)
            rotated_note = cv2.warpAffine(note, rotation_matrix, (self.notesize, self.notesize))
            masked_rotated_note = flatten_disc(rotated_note, (self.notesize / 2, self.notesize / 2), self.notesize / 2)
            rotated_templates.append(masked_rotated_note)

        candidates_with_correlation = []
        for x, y, cshift in second_grade_candidates:
            for x2 in [x - 1, x, x + 1]:
                for y2 in [y - 1, y, y + 1]:
                    possible_match = common.submatrix(self.board, x2, y2, self.notesize)
                    masked_possible_match = flatten_disc(possible_match, (self.notesize / 2, self.notesize / 2), self.notesize / 2)
                    corr = calculate_correlation(rotated_templates[cshift],
                                                 masked_possible_match,
                                                 self.settings['thresh_contrast'],
                                                 self.settings['thresh_brightness'])
                    #print 'correlation with second grade candidate at (%d,%d) is %f' % (x2, y2, corr)
                    candidates_with_correlation.append( (corr, x2, y2) )

        candidates_with_correlation.sort(reverse = True)

        final_match = candidates_with_correlation[0]

        if final_match[0] < self.settings['thresh_confidence']:
            if self.debug:
                print 'no match was found. The best was at (%d,%d) with correlation %f' % (final_match[1], final_match[2], final_match[0])
                common.qimshow(common.submatrix(self.board, final_match[1], final_match[2], self.notesize))
            return None
        else:
            if self.debug:
                print 'final match is at (%d,%d) with correlation %f' % (final_match[1], final_match[2], final_match[0])
                common.qimshow(common.submatrix(self.board, final_match[1], final_match[2], self.notesize))
                for corr, x, y in candidates_with_correlation:
                    diff_x = x - final_match[1]
                    diff_y = y - final_match[2]
                    if diff_x * diff_x + diff_y * diff_y > self.notesize * self.notesize:
                        print 'second best match is at (%d,%d) with correlation %f' % (x, y, corr)
                        common.qimshow(common.submatrix(self.board, x, y, self.notesize))
                        break

            return final_match

    def find(self, note):
        board_with_markers = cv2.cvtColor(self.board, cv2.COLOR_GRAY2BGR)
        note = cv2.cvtColor(note, cv2.COLOR_BGR2GRAY)

        first_grade_candidates = self._cifi(note)
        for x, y in first_grade_candidates:
            cv2.circle(board_with_markers, (x, y), 1, (255, 0, 255))

        if self.debug:
            common.qimshow(board_with_markers)

        second_grade_candidates = self._rafi(note, first_grade_candidates)
        for x, y, cshift in second_grade_candidates:
            #print 'best rotation for second grade candidate at %d,%d is %d' % (x, y, cshift * 10)
            cv2.circle(board_with_markers, (x, y), 3, (255, 0, 0))
            cv2.circle(board_with_markers, (x, y), 4, (255, 0, 0))

        if self.debug:
            common.qimshow(board_with_markers)

        final_match = self._tefi(note, second_grade_candidates)

        return final_match[1:]

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    board = cv2.imread('board.png')
    note = cv2.imread('note.png')

    common.qimshow([note, board], 'searching for left image in right image')

    ciratefi = Ciratefi(board, note.shape[0], debug=True)
    match = ciratefi.find(note)

