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


import cv2
import json
import math
import numpy as np
import sys
import time

import imagefuncs
import webcam

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

calibrationdata = {}

class BoardSelectionLabel(QLabel):
    def __init__(self, image, points):
        QLabel.__init__(self, None)
        self.setMouseTracking(True)

        self.originalImage = image
        self.draggedPoint = None
        self.draggablePoints = points

        self.redraw()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.draggedPoint == None:
            for i, p in enumerate(self.draggablePoints):
                if imagefuncs.eucldistance(p, (event.pos().x(), event.pos().y())) < 10:
                    self.draggedPoint = i

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if self.draggedPoint != None:
            self.updatePointPosition(self.draggedPoint, x, y)

    def mouseReleaseEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if self.draggedPoint != None:
            self.updatePointPosition(self.draggedPoint, x, y)
            self.draggedPoint = None

    def updatePointPosition(self, idx, x, y):
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        self.draggablePoints[idx] = (x,y)
        self.redraw()

    def drawImageWithCorners(self, originalImage, points):
        image = originalImage.copy();

        for p1, p2 in zip(points[:4], [points[-2]] + points):
            cv2.line(image, p1, p2, (255,0,0))
            cv2.circle(image, p1, 3, (255,0,0), 2)

        cv2.circle(image, points[-1], 10, (0,0,255), 2)
        return image

    def redraw(self):
        image_with_corners = self.drawImageWithCorners(self.originalImage, self.draggablePoints)
        pixmap = imagefuncs.cvimage_to_qpixmap(image_with_corners)
        self.setGeometry(300, 300, pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)


class BoardSelectionDialog(QDialog):

    def __init__(self, image, points, aspectratio):
        super(BoardSelectionDialog, self).__init__()

        label = BoardSelectionLabel(image, points)

        buttonBox = QDialogButtonBox(self)
        buttonBox.setGeometry(QRect(150, 250, 341, 32))
        buttonBox.setOrientation(Qt.Horizontal)
        buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        buttonBox.setObjectName("buttonBox")

        vbox = QVBoxLayout()
        vbox.addWidget(label)

        self.widthEdit = QLineEdit()
        self.widthEdit.setValidator( QDoubleValidator(0, 100, 2, self) )
        self.widthEdit.setText(str(aspectratio[0]))
        self.heightEdit = QLineEdit()
        self.heightEdit.setValidator( QDoubleValidator(0, 100, 2, self) )
        self.heightEdit.setText(str(aspectratio[1]))

        hbox = QHBoxLayout()

        hbox.addWidget(self.widthEdit)
        hbox.addWidget(self.heightEdit)
        hbox.addWidget(buttonBox)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.points = points

    def getDraggablePoints(self):
        return self.points

    def getAspectRatio(self):
        return [float(self.widthEdit.text()), float(self.heightEdit.text())]

class LaneSelectionLabel(QLabel):
    def __init__(self, image, linePositions, noteCorners):
        QLabel.__init__(self, None)
        self.setMouseTracking(True)

        self.originalImage = image
        self.draggedLine = None
        self.draggedPoint = None
        self.linePositions = linePositions
        self.noteCorners = noteCorners

        print 'original line width', self.originalImage.shape[1]

        self.redraw()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.draggedLine == None:
            for i in (0, 1):
                if imagefuncs.eucldistance(self.noteCorners[i], (event.pos().x(), event.pos().y())) < 10:
                    self.draggedPoint = i
                    return
            for i, p in enumerate(self.linePositions):
                if abs(event.pos().x() - p) < 10:
                    self.draggedLine = i
                    return

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if self.draggedPoint != None:
            self.updatePointPosition(self.draggedPoint, x, y)
        elif self.draggedLine != None:
            self.updateLinePosition(self.draggedLine, x, y)

    def mouseReleaseEvent(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if self.draggedPoint != None:
            self.updatePointPosition(self.draggedPoint, x, y)
            self.draggedPoint = None
        elif self.draggedLine != None:
            self.updateLinePosition(self.draggedLine, x, y)
            self.draggedLine = None

    def updateLinePosition(self, idx, x, y):
        if x < 0:
            x = 0
        if x >= self.originalImage.shape[1]:
            x = self.originalImage.shape[1] - 1

        self.linePositions[idx] = x
        self.redraw()

    def updatePointPosition(self, idx, x, y):
        if x < 0:
            x = 0
        if x >= self.originalImage.shape[1]:
            x = self.originalImage.shape[1] - 1

        self.noteCorners[idx] = (x, y)
        self.redraw()

    def drawImageWithLines(self, originalImage, linepositions, noteCorners):
        image = originalImage.copy();

        for linepos in linepositions:
            cv2.line(image, (linepos, 0), (linepos, image.shape[0] - 1), (255,0,0))

        cv2.rectangle(image, noteCorners[0], noteCorners[1], (0,0,255))
        cv2.circle(image, noteCorners[0], 3, (0,0,255), 2)
        cv2.circle(image, noteCorners[1], 3, (0,0,255), 2)

        return image

    def redraw(self):
        image_with_lines = self.drawImageWithLines(self.originalImage, self.linePositions, self.noteCorners)
        pixmap = imagefuncs.cvimage_to_qpixmap(image_with_lines)
        self.setGeometry(300, 300, pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)


class LaneSelectionDialog(QDialog):

    def __init__(self, image, linePositions, noteCorners):
        super(LaneSelectionDialog, self).__init__()

        label = LaneSelectionLabel(image, linePositions, noteCorners)

        buttonBox = QDialogButtonBox(self)
        buttonBox.setGeometry(QRect(150, 250, 341, 32))
        buttonBox.setOrientation(Qt.Horizontal)
        buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        buttonBox.setObjectName("buttonBox")

        vbox = QVBoxLayout()
        vbox.addWidget(label)

        vbox.addWidget(buttonBox)

        self.setLayout(vbox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.linePositions = linePositions
        self.noteCorners = noteCorners

    def getLinePositions(self):
        return sorted(self.linePositions)

    def getNoteCorners(self):
        return self.noteCorners

def loadCalibrationData():
    global calibrationdata

    try:
        with open('calibrationdata.json', 'rb') as f:
            calibrationdata = json.loads('\n'.join(f.readlines()))
    except IOError:
        # start with empty calibration data
        calibrationdata = {}

    if 'corners' not in calibrationdata:
        calibrationdata['corners'] = [ (100, 100), (200, 100), (200,300), (100,300) ]
    else:
        calibrationdata['corners'] = [ (x, y) for x, y in calibrationdata['corners'] ]

    if 'background' not in calibrationdata:
        calibrationdata['background'] = (150, 200)
    else:
        calibrationdata['background'] = tuple(calibrationdata['background'])

    if 'aspectratio' not in calibrationdata:
        calibrationdata['aspectratio'] = [1.0, 1.0]

    if 'linepositions' not in calibrationdata:
        calibrationdata['linepositions'] = []

    if 'notecorners' not in calibrationdata:
        calibrationdata['notecorners'] = []
    else:
        calibrationdata['notecorners'] = [ (x, y) for x, y in calibrationdata['notecorners'] ]

def saveCalibrationData():
    with open('calibrationdata.json', 'wb') as f:
        f.write(json.dumps(calibrationdata))

def clamp_x_positions_to_board(list_of_x, board):
    return [ min(int(x), board.shape[1]) for x in list_of_x ]

def scale_x_positions(list_of_x, scalefactor):
    return [ int(x * scalefactor) for x in list_of_x ]

def clamp_points_to_board(points, board):
    points = [ (min(int(x), board.shape[1]), min(int(y), board.shape[0])) for x,y in points ]
    return points

def scale_points(points, scalefactor):
    return [ (int(p[0] * scalefactor), int(p[1] * scalefactor)) for p in points ]

def scale_to_fit_screen(image):
    scalefactor = 1
    screen_resolution = app.desktop().screenGeometry()
    maxwidth, maxheight = screen_resolution.width(), screen_resolution.height()
    while (image.shape[0] * scalefactor > maxheight - 10 or
           image.shape[1] * scalefactor > maxwidth - 10):
        scalefactor /= 2.0

    downsized_image = cv2.resize(image, None , fx=scalefactor, fy=scalefactor)
    return downsized_image, scalefactor

if __name__ == "__main__":

    app = QApplication(sys.argv)

    loadCalibrationData()

    #
    # Let the user drag the corners and background location in the image
    #

    draggablePoints = calibrationdata['corners'] + [calibrationdata['background']]

    image = webcam.grab()

    downsized_image, scalefactor = scale_to_fit_screen(image)

    draggablePoints = scale_points(draggablePoints, scalefactor)
    draggablePoints = clamp_points_to_board(draggablePoints, downsized_image)

    dlg = BoardSelectionDialog(downsized_image, draggablePoints, calibrationdata['aspectratio'])
    if dlg.exec_() != 1:
        raise Exception('Calibration was aborted by the user')

    draggablePoints = dlg.getDraggablePoints()
    draggablePoints = scale_points(draggablePoints, 1 / scalefactor)
    draggablePoints = clamp_points_to_board(draggablePoints, image)

    calibrationdata['corners'] = draggablePoints[0:4]
    calibrationdata['background'] = draggablePoints[4]
    calibrationdata['aspectratio'] = dlg.getAspectRatio()


    #
    # Sort the corners such that the top left corner is the first in the list and the corners are in clockwise order
    #

    vertically_sorted_corners_with_index = sorted(enumerate(calibrationdata['corners']), key=lambda t: t[1][1])
    topindex, toppoint = vertically_sorted_corners_with_index[0]

    rightindex = (topindex + 1) % 4
    rightpoint = calibrationdata['corners'][rightindex]
    leftindex = (topindex + 4 - 1) % 4
    leftpoint = calibrationdata['corners'][leftindex]

    def abs_distance_per_axis(p1, p2):
        return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

    xdiff_right, ydiff_right = abs_distance_per_axis(toppoint, rightpoint)
    xdiff_left, ydiff_left = abs_distance_per_axis(toppoint, leftpoint)

    acos_left = math.acos(abs(xdiff_left) / imagefuncs.eucldistance(leftpoint, toppoint))
    acos_right = math.acos(abs(xdiff_right) / imagefuncs.eucldistance(rightpoint, toppoint))

    print('top', toppoint)
    print('left', leftpoint)
    print('right', rightpoint)

    if acos_left < acos_right:
        othertopindex = leftindex
        othertoppoint = leftpoint
    else:
        othertopindex = rightindex
        othertoppoint = rightpoint

    if toppoint[0] < othertoppoint[0]:
        startingpointindex = topindex
        direction = othertopindex - topindex
    else:
        startingpointindex = othertopindex
        direction = topindex - othertopindex

    if direction > 0:
        corners = (calibrationdata['corners'] * 2)[startingpointindex:startingpointindex + 4]
    else:
        corners = (calibrationdata['corners'] * 2)[startingpointindex + 1:startingpointindex + 4 + 1]
        corners.reverse()

    calibrationdata['corners'] = corners


    #
    # Ask the user for the aspect ratio of the scrum board (needed for perspective correction)
    #

    correctedimage, _ = imagefuncs.correct_perspective(imagefuncs.remove_color_cast(image, calibrationdata), calibrationdata, True)

    downscaled_correctedimage, scalefactor = scale_to_fit_screen(correctedimage)


    #
    # Let the user drag the lines separating todo, busy, blocked, in review and done columns
    #

    nr_of_lines = 4

    linepositions = calibrationdata['linepositions']
    noteCorners = calibrationdata['notecorners']

    linepositions = scale_x_positions(linepositions, scalefactor)
    linepositions = clamp_x_positions_to_board(linepositions, downscaled_correctedimage)

    noteCorners = scale_points(noteCorners, scalefactor)
    noteCorners = clamp_points_to_board(noteCorners, downscaled_correctedimage)

    if not linepositions:
        linepositions = sorted([(downscaled_correctedimage.shape[1] / (nr_of_lines + 1) * (i + 1)) for i in xrange(nr_of_lines)])

    if not noteCorners:
        noteCorners = [(10, 10), (50, 50)]

    dlg = LaneSelectionDialog(downscaled_correctedimage, linepositions, noteCorners)
    if dlg.exec_() != 1:
        raise Exception('Calibration was aborted by the user')

    linepositions = dlg.getLinePositions()
    linepositions = scale_x_positions(linepositions, 1 / scalefactor)
    linepositions = clamp_x_positions_to_board(linepositions, correctedimage)

    noteCorners = dlg.getNoteCorners()
    noteCorners = scale_points(noteCorners, 1 / scalefactor)
    noteCorners = clamp_points_to_board(noteCorners, correctedimage)

    calibrationdata['linepositions'] = linepositions
    calibrationdata['notecorners'] = noteCorners
    c1, c2 = calibrationdata['notecorners']
    calibrationdata['averagenotesize'] = (abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])) / 2

    saveCalibrationData()

