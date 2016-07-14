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
import imagefuncs
from findnotes import findnotes
import webcam

from PyQt5 import QtWidgets

calibrationdata = None

class NoMatchingSquareError(Exception):
    pass

def loadcalibrationdata():
    global calibrationdata

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))

class Scrumboard():
    def __init__(self, linepositions):
        self.linepositions = linepositions
        self.tasknotes = []
        self.states = ['todo', 'busy', 'blocked', 'in review', 'done']

    def to_serializable(self):
        return { 'tasknotes' : [tasknote.to_serializable() for tasknote in self.tasknotes] }

    def from_serializable(self, data):
        self.tasknotes = []
        for tasknote_data in data['tasknotes']:
            tasknote = TaskNote(None, None)
            tasknote.from_serializable(tasknote_data)
            self.tasknotes.append(tasknote)

    def get_state_from_position(self, position):
        for i, linepos in enumerate(self.linepositions):
            if position[0] < linepos:
                return self.states[i]

        return self.states[-1]

    def get_position_from_state(self, state):
        positions = [0] + self.linepositions + [None]
        stateindex = self.states.index(state)
        return positions[stateindex], positions[stateindex + 1]

    def add_tasknote(self, square):
        state = self.get_state_from_position(square.position)
        tasknote = TaskNote(square.bitmap, state)
        self.tasknotes.append(tasknote)
        return tasknote

class Square():
    def __init__(self, bitmap, position):
        self.bitmap = bitmap
        self.position = position

    def lookslike(self, otherthingwithbitmap):
        #diff = cv2.norm(self.bitmap, otherthingwithbitmap.bitmap, cv2.NORM_L1)
        #print 'lookslike calculated diff of ', diff
        # self.bitmap ~ otherthingwithbitmap.bitmap
        return False

class TaskNote():
    def __init__(self, bitmap, state):
        self.bitmap = bitmap
        self.state = state

    def to_serializable(self):
        return { 'state' : self.state,
                 'bitmap' : self.bitmap.tolist() }

    def from_serializable(self, data):
        self.state = data['state']
        self.bitmap = np.array(data['bitmap'], dtype=np.uint8)

    def setstate(self, newstate):
        self.state = newstate

    def find(self, image):
        return None

def findsquares(scrumboard, image):
    _, end_of_todo = scrumboard.get_position_from_state('todo')
    start_of_done, _ = scrumboard.get_position_from_state('done')

    masked_image = image[:,end_of_todo:start_of_done,:]
    #imagefuncs.qimshow(masked_image)
    notepositions = findnotes(masked_image)

    squares = []
    for pos in notepositions:
        notebitmap = imagefuncs.submatrix(image, pos[0] + end_of_todo, pos[1], imagefuncs.NOTE_SIZE)
        squares.append(Square(notebitmap, (pos[0] + end_of_todo, pos[1])))

        #imagefuncs.qimshow(['found square', notebitmap])

    return squares

def determine_average_colors(image):

    diameter = int(imagefuncs.NOTE_SIZE * 0.9)
    circleKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter, diameter))

    kernelSize = cv2.countNonZero(circleKernel)

    averages = cv2.filter2D(image, -1, circleKernel.astype(np.float32) / kernelSize)
    #imagefuncs.qimshow(averages)
    return averages

def readboard(previous_board_state):
    loadcalibrationdata()
    image = webcam.grab()

    #print 'Showing grabbed image'
    #imagefuncs.qimshow(image)

    correctedimage, scaled_linepositions = imagefuncs.correct_perspective(imagefuncs.remove_color_cast(image, calibrationdata), calibrationdata, False)
    #imagefuncs.qimshow(correctedimage)

    scrumboard = Scrumboard(scaled_linepositions)
    scrumboard.from_serializable(json.loads(previous_board_state))

    ciratefi = Ciratefi(correctedimage, imagefuncs.NOTE_SIZE, debug=False)

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

            newstate = scrumboard.get_state_from_position(match)
            if newstate != note.state:
                note.setstate(newstate)
                print >> sys.stderr, 'Task note found at %s. Updating state to %s' % (match, newstate)
            #imagefuncs.qimshow([ ['Task note found at (%d,%d)' % match],
            #          [imagefuncs.submatrix(correctedimage, match[0], match[1], imagefuncs.NOTE_SIZE)] ])
        else:
            print >> sys.stderr, 'Task note not found'
            unidentified_notes.append(note)

    # identify any note-sized areas of significant color on the board
    squares_in_photo = findsquares(scrumboard, maskedimage)
    for square in squares_in_photo:
        state = scrumboard.get_state_from_position(square.position)
        print >> sys.stderr, 'New task note found at %s. Setting state to %s' % (square.position, state)
        scrumboard.add_tasknote(square)

    # for any significant area of saturation that is not covered by
    # recognized squares, give a warning & highlight that it looks like a bunch of notes
    # TODO: find unused saturated areas

    averages = determine_average_colors(correctedimage)

    new_board_state = json.dumps(scrumboard.to_serializable())

    return new_board_state

if __name__ == "__main__":
    scrumboardfile = 'scrumboardstate.json'
    app = QtWidgets.QApplication(sys.argv)

    with open(scrumboardfile, 'w+') as f:
        old_state = f.read()
        new_state = readboard(old_state)
        f.write(new_state)


