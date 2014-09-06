#!/usr/bin/env python

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

from gi.repository import Gtk
import json
import math
import numpy as np
import cv2

wndname = "Calibration";

calibrationdata = {}
draggablePoints = []
draggedPoint = None
linepositions = []
draggedLine = None

def eucldistance(p1, p2):
    return cv2.norm(np.array(p1) - np.array(p2))

def drawImageWithCorners(originalImage):
    image = originalImage.copy();

    for p1, p2 in zip(draggablePoints[:4], [draggablePoints[-2]] + draggablePoints):
        cv2.line(image, p1, p2, (255,0,0))
        cv2.circle(image, p1, 3, (255,0,0), 2)

    cv2.circle(image, draggablePoints[-1], 10, (0,0,255), 2)

    cv2.imshow(wndname, image)

def mouseHandler1(event, x, y, flags, param):
    global draggablePoints, draggedPoint

    originalImage = param

    # user press left button
    if event == cv2.EVENT_LBUTTONDOWN and draggedPoint == None:
        for i, p in enumerate(draggablePoints):
            if eucldistance(p, (x,y)) < 10:
                draggedPoint = i

    # user drag the mouse
    if ( (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONUP) and draggedPoint != None):

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        draggablePoints[draggedPoint] = (x,y)

        drawImageWithCorners(originalImage)

        # user release left button
        if event == cv2.EVENT_LBUTTONUP:
            draggedPoint = None

def drawImageWithLines(originalImage):
    image = originalImage.copy();

    for linepos in linepositions:
        cv2.line(image, (linepos, 0), (linepos, image.shape[1] - 1), (255,0,0))

    cv2.imshow(wndname, image)

def mouseHandler2(event, x, y, flags, param):
    global linepositions, draggedLine

    originalImage = param

    # user press left button
    if event == cv2.EVENT_LBUTTONDOWN and draggedLine == None:
        for i, linepos in enumerate(linepositions):
            if abs(x - linepos) < 10:
                draggedLine = i

    # user drag the mouse
    if ( (event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONUP) and draggedLine != None):

        if x < 0:
            x = 0
        if x >= originalImage.shape[0]:
            x = originalImage.shape[0] - 1

        linepositions[draggedLine] = x

        drawImageWithLines(originalImage)

        # user release left button
        if event == cv2.EVENT_LBUTTONUP:
            draggedLine = None

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
        calibrationdata['linepositions'] = None

def saveCalibrationData():
    with open('calibrationdata.json', 'wb') as f:
        f.write(json.dumps(calibrationdata))

if __name__ == "__main__":
    cv2.namedWindow( wndname, 1 );
    cv2.moveWindow(wndname, 100, 100);

    loadCalibrationData()

    #
    # Let the user drag the corners and background location in the image
    #

    draggablePoints = calibrationdata['corners'] + [calibrationdata['background']]

    image = cv2.imread("webcam.jpg", 1);

    drawImageWithCorners(image);

    cv2.setMouseCallback(wndname, mouseHandler1, image)

    while 1:
        k = cv2.waitKey(1) & 0xFF;
        if k == 27:
            break


    calibrationdata['corners'] = draggablePoints[0:4]
    calibrationdata['background'] = draggablePoints[4]


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

    acos_left = math.acos(abs(xdiff_left) / eucldistance(leftpoint, toppoint))
    acos_right = math.acos(abs(xdiff_right) / eucldistance(rightpoint, toppoint))

    print 'top', toppoint
    print 'left', leftpoint
    print 'right', rightpoint

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
    # Ask the user for the aspect ratio of the scrumboard (needed for perspective correction)
    #


    win = Gtk.Window()
    win.connect("delete-event", Gtk.main_quit)

    vbox = Gtk.VBox(False, 0)
    win.add(vbox)

    entry1 = Gtk.Entry()
    entry1.set_text(str(calibrationdata['aspectratio'][0]))
    vbox.pack_start(entry1, True, True, 0)

    entry2 = Gtk.Entry()
    entry2.set_text(str(calibrationdata['aspectratio'][1]))
    vbox.pack_start(entry2, True, True, 0)

    def buttonclicked(window):
        calibrationdata['aspectratio'] = [float(entry1.get_text()), float(entry2.get_text())]
        win.hide()
        Gtk.main_quit()

    button = Gtk.Button(stock=Gtk.STOCK_CLOSE)
    button.connect("clicked", buttonclicked)
    vbox.pack_start(button, True, True, 0)

    win.show_all()
    Gtk.main()


    width = 1000
    height = int(width * calibrationdata['aspectratio'][1] / calibrationdata['aspectratio'][0])

    print 'width: ', width
    print 'height: ', height

    correctedrectangle = np.array([(0,0), (width, 0), (width, height), (0, height)], np.float32)

    orderedcorners = np.array(calibrationdata['corners'], np.float32)
    transformation = cv2.getPerspectiveTransform(orderedcorners, correctedrectangle)
    correctedimage = cv2.warpPerspective(image, transformation, (width, height))



    #
    # Let the user drag the lines separating todo, busy, blocked, in review and done columns
    #

    nr_of_lines = 4

    linepositions = calibrationdata['linepositions']

    if not linepositions:
        linepositions = sorted([(correctedimage.shape[0] / (nr_of_lines + 1) * (i + 1)) for i in xrange(nr_of_lines)])

    drawImageWithLines(correctedimage);

    cv2.setMouseCallback(wndname, mouseHandler2, correctedimage)

    while 1:
        k = cv2.waitKey(1) & 0xFF;
        if k == 27:
            break


    calibrationdata['linepositions'] = linepositions

    saveCalibrationData()

