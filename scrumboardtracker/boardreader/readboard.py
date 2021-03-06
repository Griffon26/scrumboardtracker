#!/usr/bin/env python2

# Copyright 2014-2016 Maurice van der Pot <griffon26@kfk4ever.com>
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
import numpy as np
import os
import sys

from ciratefi import Ciratefi
import scrumboardtracker.board as board
import imagefuncs
from findnotes import findnotes

from PyQt5 import QtWidgets

calibrationdata = None

class NoMatchingSquareError(Exception):
    pass

def loadcalibrationdata():
    global calibrationdata

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))

def findsquares(scrumboard, image):
    start_of_done, _ = scrumboard.get_position_from_state('done')

    masked_image = image[:,:start_of_done,:]
    #imagefuncs.qimshow(masked_image)
    notepositions = findnotes(masked_image)

    squares = []
    for pos in notepositions:
        notebitmap = imagefuncs.submatrix(image, pos[0], pos[1], imagefuncs.NOTE_SIZE)
        squares.append(board.Square(notebitmap, (pos[0], pos[1])))

        #imagefuncs.qimshow(['found square', notebitmap])

    return squares

def determine_average_colors(image):

    diameter = int(imagefuncs.NOTE_SIZE * 0.9)
    circleKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter, diameter))

    kernelSize = cv2.countNonZero(circleKernel)

    averages = cv2.filter2D(image, -1, circleKernel.astype(np.float32) / kernelSize)
    #imagefuncs.qimshow(averages)
    return averages

def get_scaled_linepositions(calibdata):
    scale = (1.0 * imagefuncs.NOTE_SIZE) / calibdata['averagenotesize']
    scaled_linepositions = [int(l * scale) for l in calibdata['linepositions']]
    return scaled_linepositions

def get_coordinate_order(coords):
    maxdiff = imagefuncs.NOTE_SIZE / 2
    order = {}
    currentclass = None
    for c in sorted(coords):
        if not currentclass or c > currentclass + maxdiff:
            currentclass = c
        order[c] = currentclass

    return order

def sort_squares_by_approximate_position(squares):
    x_coords, y_coords = zip(*(s.position for s in squares))
    x_order = get_coordinate_order(x_coords)
    y_order = get_coordinate_order(y_coords)

    def sortkey(square):
        return (y_order[square.position[1]], x_order[square.position[0]])

    return sorted(squares, key=sortkey)

def readboard(previous_board_state, imagefile=None, debug=False):
    loadcalibrationdata()

    scaled_linepositions = get_scaled_linepositions(calibrationdata)

    scrumboard = board.Scrumboard(scaled_linepositions)
    if len(previous_board_state):
        scrumboard.from_serializable(json.loads(previous_board_state))

    if imagefile:
        correctedimage = cv2.imread(imagefile)
        #imagefuncs.qimshow(correctedimage)
        scrumboard.set_bitmap(correctedimage)
    else:
        correctedimage = scrumboard.get_bitmap()

    ciratefi = Ciratefi(correctedimage, imagefuncs.NOTE_SIZE, debug=debug)

    maskedimage = correctedimage.copy()

    # find all known notes on the board and update their states
    # - not finding them is ok if previously in todo or in done
    # - otherwise warn about missing note (not implemented yet)
    unidentified_notes = []
    for note in scrumboard.tasknotes:
        #print 'Searching for task note previously in state %s' % note.state
        #imagefuncs.qimshow([ ['Searching for task note previously in state %s' % note.state],
        #          [note.bitmap] ])
        match = ciratefi.find(note.bitmap)

        if match:
            imagefuncs.masksubmatrix(maskedimage, match[0], match[1], imagefuncs.NOTE_SIZE)

            oldstate = note.state
            oldposition = note.position
            newstate = scrumboard.move_tasknote(note, match)
            if newstate != oldstate:
                print >> sys.stderr, 'Task note %d found at %s (previously at %s), updating state from %s to %s' % (note.taskid, match, oldposition, oldstate, newstate)
            else:
                print >> sys.stderr, 'Task note %d found at %s (previously at %s), so state is still %s' % (note.taskid, match, oldposition, newstate)
            #imagefuncs.qimshow([ ['Task note found at (%d,%d)' % match],
            #          [imagefuncs.submatrix(correctedimage, match[0], match[1], imagefuncs.NOTE_SIZE)] ])
        else:
            print >> sys.stderr, 'Task note %d (previously at %s in state %s) not found' % (note.taskid, note.position, note.state)
            unidentified_notes.append(note)

    # identify any note-sized areas of significant color on the board
    squares_in_photo = findsquares(scrumboard, maskedimage)
    squares_in_photo = sort_squares_by_approximate_position(squares_in_photo)
    for square in squares_in_photo:
        state = scrumboard.get_state_from_position(square.position)
        tasknote = scrumboard.add_tasknote(square)
        print >> sys.stderr, 'New task note found at %s. New task %d created with state %s' % (tasknote.position, tasknote.taskid, tasknote.state)

    # for any significant area of saturation that is not covered by
    # recognized squares, give a warning & highlight that it looks like a bunch of notes
    # TODO: find unused saturated areas

    averages = determine_average_colors(correctedimage)

    new_board_state = json.dumps(scrumboard.to_serializable())

    return new_board_state

