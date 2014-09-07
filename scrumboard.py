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

import json
import numpy as np
import cv2

MINIMUM_NOTE_SIZE = 40

wndname = "Scrumboard"
calibrationdata = None

def waitforescape():
    while 1:
        k = cv2.waitKey(1) & 0xFF;
        if k == 27:
            break


def loadimage():
    image = cv2.imread("webcam.jpg", 1);
    cv2.imshow(wndname, image)
    #waitforescape()

    return image


def remove_color_cast(image):
    bgr_planes = cv2.split(image)

    backgroundpos = calibrationdata['background']
    average_bgr = image[backgroundpos[0]][backgroundpos[1]]
    print 'bgr of scrumboard background:', average_bgr

    for i in xrange(3):
        thresh = average_bgr[i]
        plane = bgr_planes[i]
        print "Showing plane"
        cv2.imshow(wndname, plane);
        #waitforescape()

        retval, mask = cv2.threshold(plane, thresh, 255, cv2.THRESH_BINARY);
        print "Showing mask"
        cv2.imshow(wndname, mask)
        #waitforescape()

        highvalues = (plane - thresh) * (128.0 / (255 - thresh)) + 128;
        highvalues = highvalues.astype(np.uint8)

        highvalues_masked = cv2.bitwise_and(highvalues, mask)
        print "Showing scaled high values"
        cv2.imshow(wndname, highvalues_masked)
        #waitforescape()

        mask = 255 - mask;
        lowvalues = cv2.bitwise_and(plane, mask)
        print "Showing low values"
        cv2.imshow(wndname, lowvalues)
        #waitforescape()

        lowvalues = lowvalues * 128.0 / thresh
        lowvalues = lowvalues.astype(np.uint8)
        print "Showing scaled low values"
        cv2.imshow(wndname, lowvalues)
        #waitforescape()

        bgr_planes[i] = lowvalues + highvalues_masked
        print "Showing scaled plane"
        cv2.imshow(wndname, bgr_planes[i])
        #waitforescape()

    correctedimage = cv2.merge(bgr_planes)
    correctedimage = correctedimage.astype(np.uint8)
    print "Showing corrected image"
    cv2.imshow(wndname, correctedimage)
    #waitforescape()

    return correctedimage

def loadcalibrationdata():
    global calibrationdata

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))

class Scrumboard():
    def __init__(self, linepositions):
        self.linepositions = linepositions
        self.tasknotes = set()
        self.states = ['todo', 'busy', 'blocked', 'in review', 'done']

    def load_state_from_file(self):
        pass

    def save_state_to_file(self):
        pass

    def get_state_from_position(self, position):
        for i, linepos in enumerate(linepositions[0]):
            if position[0] < linepos:
                return self.states[i]

        return self.states[-1]

    def add_tasknote(square):
        state = self.get_state_from_position(square.position)
        tasknote = TaskNote(square.bitmap, state)
        self.tasknotes.add(tasknote)

class Square():
    def __init__(self):
        self.bitmap = None

    def lookslike(otherthingwithbitmap):
        # self.bitmap ~ otherthingwithbitmap.bitmap
        return False

class TaskNote():
    def __init__(self, bitmap, state):
        self.bitmap = bitmap
        self.state = state

    def setstate(newstate):
        self.state = newstate

def flatten(list_with_sublists):
    return [item for sublist in list_with_sublists for item in sublist]

def correct_perspective(image):
    width = 1000
    height = int(width * calibrationdata['aspectratio'][1] / calibrationdata['aspectratio'][0])

    print 'width: ', width
    print 'height: ', height

    correctedrectangle = np.array([(0,0), (width, 0), (width, height), (0, height)], np.float32)

    orderedcorners = np.array(calibrationdata['corners'], np.float32)
    transformation = cv2.getPerspectiveTransform(orderedcorners, correctedrectangle)
    correctedimage = cv2.warpPerspective(image, transformation, (width, height))

    return correctedimage

def findsquares(image):
    denoised = cv2.pyrUp(cv2.pyrDown(image))

    hsv = cv2.cvtColor( denoised, cv2.COLOR_BGR2HSV )
    _, saturation, _ = cv2.split(hsv)

    color_only = cv2.inRange(saturation, 25, 255)

    kernel = np.ones((3,3), np.uint8)
    color_only = cv2.morphologyEx(color_only, cv2.MORPH_CLOSE, kernel)
    color_only = cv2.morphologyEx(color_only, cv2.MORPH_OPEN, kernel)
    cv2.imshow(wndname, color_only)
    waitforescape()


    colorless_only = 255 - color_only
    distance_to_color = cv2.distanceTransform(colorless_only, cv2.cv.CV_DIST_L2, 3)

    normdist = cv2.normalize(distance_to_color, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow(wndname, normdist)
    waitforescape()

    _, sure_bg = cv2.threshold(distance_to_color, 20, 1, cv2.THRESH_BINARY)

    cv2.imshow(wndname, sure_bg)
    waitforescape()

    distance_to_colorless = cv2.distanceTransform(color_only, cv2.cv.CV_DIST_L2, 3)

    normdist = cv2.normalize(distance_to_colorless, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow(wndname, normdist)
    waitforescape()

    _, sure_fg = cv2.threshold(distance_to_colorless, 20, 1, cv2.THRESH_BINARY)
    cv2.imshow(wndname, sure_fg)
    waitforescape()

    sure_fg8 = np.uint8(sure_fg)

    contours, hierarchy = cv2.findContours(sure_fg8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_comps = len(contours)
    markers = np.zeros(sure_fg.shape, np.int32)
    for i in xrange(n_comps):
        cv2.drawContours(markers, contours, i, i + 1, -1)
    markers[sure_bg == 0] = n_comps + 1

    normmarkers = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow(wndname, normmarkers)
    #waitforescape()

    cv2.watershed(hsv, markers)

    normmarkers = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow(wndname, normmarkers)
    #waitforescape()

    component_contours = []
    for i in xrange(n_comps):
        singlecomponent = cv2.inRange(markers, i, i)
        contours = cv2.findContours(singlecomponent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            if len(contours) > 1:
                # TODO: check if ignoring is the right thing to do
                raise RuntimeError("More than one (%d) external contour found for component %d. Not sure what to do with this. Ignoring component for now." % (len(contours), i))
            else:
                component_contours.append(contours[0])


    return []

if __name__ == "__main__":
    cv2.namedWindow( wndname, 1 );
    cv2.moveWindow(wndname, 100, 100);

    loadcalibrationdata()
    image = loadimage()
    correctedimage = correct_perspective(remove_color_cast(image))

    # The rest of this program should:
    # - read the known state of the board from file:
    #   (consists of bitmaps for all known notes and last known state for each)
    # - identify any squares on the board
    # - take their bitmaps and compare them to all known bitmaps
    #   - for all matches found, determine which column they are in based on position and update their state
    #   - any remaining bitmaps are new ones, store the bitmap and the state
    # - go through the remaining unmatched bitmaps and find them on the board
    #   - for all matches found, determine which column they are in based on position and update their state
    #   - any remaining bitmaps are notes that are no longer visible:
    #     - if the note was Done/Todo before, assume it's still Done/Todo
    #     - otherwise, give a warning & highlight
    # - for any significant area of saturation that is not covered by
    #   recognized squares, give a warning & highlight that it looks like a bunch of notes

    scrumboard = Scrumboard(calibrationdata['linepositions'])
    scrumboard.load_state_from_file()

    squares_in_photo = findsquares(correctedimage)
    matches_per_square = []

    for square in squares_in_photo:
        matching_tasknotes = []
        for tasknote in scrumboard.tasknotes:
            if square.lookslike(tasknote):
                matching_tasknotes.append(tasknote)
        matches_per_square.append(matching_tasknotes)

    for square, matches in zip(squares_in_photo, matches_per_square):
        if len(matches) > 1:
            raise RuntimeError('More than one task note matched a square')
        elif len(matches) == 1:
            newstate = scrumboard.get_state_from_position(square.position)
            matches[0].setstate(newstate)
        else: # len(matches) == 0
            scrumboard.add_tasknote(square)

    for tasknote in scrumboard.tasknotes:
        if tasknote not in set(flatten(matches_per_square)):
            position = tasknote.find(correctedimage)
            if position:
                newstate = scrumboard.get_state_from_position(position)
                tasknote.setstate(newstate)
            else:
                if tasknote.state != 'done' and tasknote.state != 'todo':
                    raise RuntimeError("Can't find note and it wasn't in todo/done before")

    # TODO: find unused saturated areas

    scrumboard.save_state_to_file()



