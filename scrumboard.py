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



#
# The purpose of this program is to monitor a physical scrum board using a
# webcam, identify task notes and track how they move across the board over
# time. It does not need to read text on the notes, it only needs to recognize
# each note as distinct from all the other notes.
#
# It should use the information about tasks to interface with a digital
# scrumboard such as the one provided by jira and keep task states in sync with
# the physical board.
#
# In order to interact with the digital scrumboard the application must know
# which physical note corresponds to which task in e.g. jira. I haven't chosen
# yet how to deal with this. I think the easiest way from a user perspective
# would be to ask for the JIRA task ID whenever the task is first seen on the
# board (i.e. identified as a square that doesn't match with any previously
# seen squares). An alternative could be a step at the start of the sprint to
# set up (assisted by the user) the initial links between physical notes and
# digital tasks.
#

import copy
import cv2
import json
import numpy as np
import os
import sys

from ciratefi import Ciratefi
import common
from common import qimshow
from findnotes import findnotes
import webcam

from PyQt5 import QtWidgets

wndname = "Scrumboard"
calibrationdata = None

class NoMatchingSquareError(Exception):
    pass

def waitforescape():
    while 1:
        k = cv2.waitKey(1) & 0xFF;
        if k == 27:
            break


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

def loadcalibrationdata():
    global calibrationdata

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))

class Scrumboard():
    def __init__(self, linepositions):
        self.linepositions = linepositions
        self.tasknotes = []
        self.states = ['todo', 'busy', 'blocked', 'in review', 'done']

    def load_state_from_file(self):
        if os.path.exists('scrumboardstate.json'):
            with open('scrumboardstate.json', 'rb') as f:
                self.from_serializable(json.load(f))

    def save_state_to_file(self):
        with open('scrumboardstate.json', 'wb') as f:
            f.write(json.dumps(self.to_serializable()))

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

def flatten(list_with_sublists):
    return [item for sublist in list_with_sublists for item in sublist]

def findsquares(scrumboard, image):
    _, end_of_todo = scrumboard.get_position_from_state('todo')
    start_of_done, _ = scrumboard.get_position_from_state('done')

    masked_image = image[:,end_of_todo:start_of_done,:]
    qimshow(masked_image)
    notepositions = findnotes(masked_image)

    squares = []
    for pos in notepositions:
        notebitmap = common.submatrix(image, pos[0] + end_of_todo, pos[1], common.NOTE_SIZE)
        squares.append(Square(notebitmap, (pos[0] + end_of_todo, pos[1])))

        #qimshow(['found square', notebitmap])

    return squares


def find_largest_overlapping_square(singlecontour, imagecutout):

    overallMax = 0

    newcenter = (singlecontour.shape[0] / 2, singlecontour.shape[1] / 2)

    for angle in xrange(-16,17):
        rotation = cv2.getRotationMatrix2D(newcenter, angle, 1.0)
        rotatedcontour = cv2.warpAffine(singlecontour, rotation, singlecontour.shape)

        #qimshow(rotatedcontour)

        offsetcontour = rotatedcontour.astype(np.float32) - 160

        for kernel_size in [common.NOTE_SIZE]: #xrange(common.NOTE_SIZE, int(min(imagecutout.shape[0], common.NOTE_SIZE * 1.1))):
            boxfiltered = cv2.boxFilter(offsetcontour, -1, (kernel_size, kernel_size), None, (0,0), False)

            norm = cv2.normalize(boxfiltered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            #qimshow(norm)


            mask = np.zeros(boxfiltered.shape, np.uint8)
            mask[0:boxfiltered.shape[0] - kernel_size + 1, 0:boxfiltered.shape[1] - kernel_size + 1] = 1

            minVal, maxVal, _, maxLoc = cv2.minMaxLoc(boxfiltered, mask)

            #print 'max is %s at angle %s and size %s' % (maxVal, angle, kernel_size)
            if maxVal > overallMax:
                overallMax = maxVal
                overallMaxLoc = maxLoc
                overallMaxAngle = angle
                overallMaxKernelSize = kernel_size

    bitmap = None
    if overallMax > 0 and overallMaxKernelSize < common.NOTE_SIZE * 1.1:
        rotation = cv2.getRotationMatrix2D(newcenter, overallMaxAngle, 1.0)
        rotatedimagecutout = cv2.warpAffine(imagecutout, rotation, imagecutout.shape[0:2])
        bitmap = rotatedimagecutout[overallMaxLoc[0]:overallMaxLoc[0] + overallMaxKernelSize, overallMaxLoc[1]:overallMaxLoc[1] + overallMaxKernelSize]

        print 'showing largest overlapping square with score %s' % overallMax
        #qimshow(bitmap)
    else:
        raise NoMatchingSquareError()

    return -overallMaxAngle, overallMaxLoc, overallMaxKernelSize, bitmap

def determine_average_colors(image):

    diameter = int(common.NOTE_SIZE * 0.9)
    circleKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter, diameter))

    kernelSize = cv2.countNonZero(circleKernel)

    averages = cv2.filter2D(image, -1, circleKernel.astype(np.float32) / kernelSize)
    #qimshow(averages)
    return averages

# The basic algorithm for updating the scrumboard state from an image is this:
# - find all known notes on the board and update their states
#   - not finding them is ok if previously in todo or in done
#   - otherwise warn about missing note
# - identify squares, filter out all at positions of recognized known notes
# - add remaining squares as new notes, show new notes to user
# - warn about significant areas with color (outside todo/done) not covered by notes
if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    loadcalibrationdata()
    image = webcam.grab()

    print 'showing grabbed image'
    #qimshow(image)

    correctedimage, scaled_linepositions = common.correct_perspective(common.remove_color_cast(image, calibrationdata), calibrationdata, False)
    qimshow(correctedimage)

    # The rest of this program should:

    scrumboard = Scrumboard(scaled_linepositions)

    # read the known state of the board from file:
    # (consists of bitmaps for all known notes and last known state for each)
    scrumboard.load_state_from_file()

    ciratefi = Ciratefi(correctedimage, common.NOTE_SIZE, debug=False)

    maskedimage = correctedimage.copy()

    unidentified_notes = []
    for note in scrumboard.tasknotes:
        print 'Searching for task note previously in state %s' % note.state
        #qimshow([ ['Searching for task note previously in state %s' % note.state],
        #          [note.bitmap] ])
        match = ciratefi.find(note.bitmap)

        if match:
            common.masksubmatrix(maskedimage, match[0], match[1], common.NOTE_SIZE)

            newstate = scrumboard.get_state_from_position(match)
            if newstate != note.state:
                note.setstate(newstate)
                print 'Task note found at %s. Updating state to %s' % (match, newstate)
            #qimshow([ ['Task note found at (%d,%d)' % match],
            #          [common.submatrix(correctedimage, match[0], match[1], common.NOTE_SIZE)] ])
        else:
            print 'Task note not found'
            unidentified_notes.append(note)

    # identify any squares on the board
    squares_in_photo = findsquares(scrumboard, maskedimage)
    for square in squares_in_photo:
        state = scrumboard.get_state_from_position(square.position)
        print 'New task note found at %s. Setting state to %s' % (square.position, state)
        scrumboard.add_tasknote(square)

    # for any significant area of saturation that is not covered by
    # recognized squares, give a warning & highlight that it looks like a bunch of notes
    # TODO: find unused saturated areas

    averages = determine_average_colors(correctedimage)

    scrumboard.save_state_to_file()



