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

import common
import webcam

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

wndname = "Scrumboard"
calibrationdata = None

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

        vbox = QVBoxLayout()

        if text != None:
            vbox.addWidget(QLabel(text))

        hbox = QHBoxLayout()
        for image in images:
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
        return { 'linepositions' : self.linepositions,
                 'tasknotes' : [tasknote.to_serializable() for tasknote in self.tasknotes] }

    def from_serializable(self, data):
        self.linepositions = data['linepositions']
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

    def setstate(newstate):
        self.state = newstate

    def find(self, image):
        return None

def flatten(list_with_sublists):
    return [item for sublist in list_with_sublists for item in sublist]

def findsquares(image):
    print 'showing image to search for squares'
    qimshow(image)

    denoised = cv2.pyrUp(cv2.pyrDown(image))

    hsv = cv2.cvtColor( denoised, cv2.COLOR_BGR2HSV )
    _, saturation, _ = cv2.split(hsv)
    print 'showing saturation'
    #qimshow(saturation)

    color_only = cv2.inRange(saturation, 25, 255)

    kernel = np.ones((2,2), np.uint8)
    color_only = cv2.morphologyEx(color_only, cv2.MORPH_CLOSE, kernel)
    color_only = cv2.morphologyEx(color_only, cv2.MORPH_OPEN, kernel)
    print 'showing color_only'
    #qimshow(color_only)


    colorless_only = 255 - color_only
    distance_to_color = cv2.distanceTransform(colorless_only, cv2.DIST_L2, 3)

    normdist = cv2.normalize(distance_to_color, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    print 'showing distance_to_color'
    #qimshow(normdist)

    _, sure_bg = cv2.threshold(distance_to_color, common.NOTE_SIZE / 2.3, 1, cv2.THRESH_BINARY)

    print 'showing sure_bg'
    #qimshow(sure_bg)

    distance_to_colorless = cv2.distanceTransform(color_only, cv2.DIST_L2, 3)

    normdist = cv2.normalize(distance_to_colorless, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #print 'showing distance_to_colorless'
    #qimshow(normdist)

    _, sure_fg = cv2.threshold(distance_to_colorless, common.NOTE_SIZE / 2.3, 1, cv2.THRESH_BINARY)
    #print 'showing sure_fg'
    #qimshow(sure_fg)

    sure_fg8 = np.uint8(sure_fg)

    _, contours, _ = cv2.findContours(sure_fg8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_comps = len(contours)
    markers = np.zeros(sure_fg.shape, np.int32)
    for i in xrange(n_comps):
        cv2.drawContours(markers, contours, i, i + 1, -1)
    markers[sure_bg == 1] = n_comps + 1

    normmarkers = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #print 'showing markers before watershed'
    #qimshow(normmarkers)

    watershed_input = cv2.cvtColor(saturation,cv2.COLOR_GRAY2RGB);
    cv2.watershed(watershed_input, markers)

    normmarkers = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #print 'showing markers after watershed'
    #qimshow(normmarkers)

    squares = []
    for i in xrange(n_comps):
        singlecomponent = cv2.inRange(markers, i, i)

        _, contours, _ = cv2.findContours(singlecomponent.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        norm = cv2.normalize(singlecomponent, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        print 'showing single component %d' % i
        #qimshow(norm)

        if len(contours) != 0:
            if len(contours) > 1:
                # TODO: check if ignoring is the right thing to do
                raise RuntimeError ("More than one (%d) external contour found for component %d. Not sure what to do with this. Ignoring component for now." % (len(contours), i))
            else:
                contour = contours[0]

                center, radius = cv2.minEnclosingCircle(contour)
                size = (int(radius) * 2, int(radius) * 2)

                singlecontour = cv2.getRectSubPix(singlecomponent, size, center)
                imagecutout = cv2.getRectSubPix(image, size, center)

                print 'showing imagecutout'
                #qimshow(singlecontour)

                try:
                    angle, topleft, size, square_bitmap = find_largest_overlapping_square(singlecontour, imagecutout)
                except NoMatchingSquareError:
                    # continue with the next component
                    continue
                squares.append(Square(square_bitmap, center))

                x = topleft[0]
                y = topleft[1]
                frame = np.zeros(singlecontour.shape, np.uint8)
                cv2.rectangle(frame, (x,y), (x + size, y + size), 255, 1)
                #qimshow(frame)

                rotation = cv2.getRotationMatrix2D( (frame.shape[0] / 2, frame.shape[1] / 2), angle, 1.0)
                rotatedframe = cv2.warpAffine(frame, rotation, frame.shape)
                #qimshow(rotatedframe)

                print 'showing imagecutout'
                #qimshow(imagecutout)

                rotatedframe = cv2.cvtColor( rotatedframe, cv2.COLOR_GRAY2BGR )
                cutout_with_frame = cv2.addWeighted(imagecutout, 1.0, rotatedframe, 0.5, 0)
                print 'showing imagecutout with frame'
                #qimshow(cutout_with_frame)

    return squares


def find_largest_overlapping_square(singlecontour, imagecutout):

    overallMax = 0

    newcenter = (singlecontour.shape[0] / 2, singlecontour.shape[1] / 2)

    print('imagecutout.shape', imagecutout.shape)
    #qimshow(imagecutout)

    for angle in xrange(-16,17):
        rotation = cv2.getRotationMatrix2D(newcenter, angle, 1.0)
        rotatedcontour = cv2.warpAffine(singlecontour, rotation, singlecontour.shape)

        #qimshow(rotatedcontour)

        offsetcontour = rotatedcontour.astype(np.float32) - 160

        for kernel_size in xrange(common.NOTE_SIZE, int(min(imagecutout.shape[0], common.NOTE_SIZE * 1.1))):
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


if __name__ == "__main__":

    app = QApplication(sys.argv)

    loadcalibrationdata()
    image = webcam.grab()

    print 'showing grabbed image'
    #qimshow(image)

    correctedimage = common.correct_perspective(remove_color_cast(image), calibrationdata, False)

    # The rest of this program should:

    scrumboard = Scrumboard(calibrationdata['linepositions'])

    # read the known state of the board from file:
    # (consists of bitmaps for all known notes and last known state for each)
    scrumboard.load_state_from_file()

    for note in scrumboard.tasknotes:
        print 'Showing task note in state %s' % note.state
        #qimshow(note.bitmap)


    # identify any squares on the board
    squares_in_photo = findsquares(correctedimage)

    # make a copy of the list of previously known task notes before we start adding new ones
    previously_known_tasknotes = copy.copy(scrumboard.tasknotes)

    # take the bitmaps of the squares in the photo and compare them to the bitmaps of all task notes that we know of
    matches_per_square = []
    for square in squares_in_photo:
        matching_tasknotes = []
        for tasknote in scrumboard.tasknotes:
            if square.lookslike(tasknote):
                matching_tasknotes.append(tasknote)
        matches_per_square.append(matching_tasknotes)

    # for all matches found, determine which column they are in based on position and update their state
    # any remaining bitmaps are new ones, store the bitmap and the state
    for square, matches in zip(squares_in_photo, matches_per_square):

        if len(matches) > 1:
            raise RuntimeError('More than one task note matched a square')
        elif len(matches) == 1:
            newstate = scrumboard.get_state_from_position(square.position)
            matches[0].setstate(newstate)
            qimshow([square.bitmap] + matches, 'Square with matching notes')
        else: # len(matches) == 0
            scrumboard.add_tasknote(square)

    # go through the previously known task notes that weren't matched yet and find them on the board
    for tasknote in previously_known_tasknotes:
        if tasknote not in set(flatten(matches_per_square)):
            position = tasknote.find(correctedimage)
            # for all matches found, determine which column they are in based on position and update their state
            if position:
                newstate = scrumboard.get_state_from_position(position)
                tasknote.setstate(newstate)
            # any remaining bitmaps are notes that are no longer visible:
            else:
                # if the note was Done/Todo before, assume it's still Done/Todo
                # otherwise, give a warning & highlight
                if tasknote.state != 'done' and tasknote.state != 'todo':
                    #raise RuntimeError("Can't find note and it wasn't in todo/done before")
                    print "Can't find note and it wasn't in todo/done before. Ignoring; note state not updated."

    # for any significant area of saturation that is not covered by
    # recognized squares, give a warning & highlight that it looks like a bunch of notes
    # TODO: find unused saturated areas

    scrumboard.save_state_to_file()



