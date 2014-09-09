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
    def __init__(self, bitmap):
        self.bitmap = bitmap

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
    print 'showing image to search for squares'
    cv2.imshow(wndname, image)
    waitforescape()

    denoised = cv2.pyrUp(cv2.pyrDown(image))

    hsv = cv2.cvtColor( denoised, cv2.COLOR_BGR2HSV )
    _, saturation, _ = cv2.split(hsv)

    color_only = cv2.inRange(saturation, 25, 255)

    kernel = np.ones((3,3), np.uint8)
    color_only = cv2.morphologyEx(color_only, cv2.MORPH_CLOSE, kernel)
    color_only = cv2.morphologyEx(color_only, cv2.MORPH_OPEN, kernel)
    print 'showing color_only'
    cv2.imshow(wndname, color_only)
    #waitforescape()


    colorless_only = 255 - color_only
    distance_to_color = cv2.distanceTransform(colorless_only, cv2.cv.CV_DIST_L2, 3)

    normdist = cv2.normalize(distance_to_color, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    print 'showing distance_to_color'
    cv2.imshow(wndname, normdist)
    #waitforescape()

    _, sure_bg = cv2.threshold(distance_to_color, 20, 1, cv2.THRESH_BINARY)

    print 'showing sure_bg'
    cv2.imshow(wndname, sure_bg)
    #waitforescape()

    distance_to_colorless = cv2.distanceTransform(color_only, cv2.cv.CV_DIST_L2, 3)

    normdist = cv2.normalize(distance_to_colorless, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    print 'showing distance_to_colorless'
    cv2.imshow(wndname, normdist)
    #waitforescape()

    _, sure_fg = cv2.threshold(distance_to_colorless, 20, 1, cv2.THRESH_BINARY)
    print 'showing sure_fg'
    cv2.imshow(wndname, sure_fg)
    #waitforescape()

    sure_fg8 = np.uint8(sure_fg)

    contours, hierarchy = cv2.findContours(sure_fg8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_comps = len(contours)
    markers = np.zeros(sure_fg.shape, np.int32)
    for i in xrange(n_comps):
        cv2.drawContours(markers, contours, i, i + 1, -1)
    markers[sure_bg == 1] = n_comps + 1

    normmarkers = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    print 'showing markers before watershed'
    cv2.imshow(wndname, normmarkers)
    #waitforescape()

    cv2.watershed(hsv, markers)

    normmarkers = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    print 'showing markers after watershed'
    cv2.imshow(wndname, normmarkers)
    waitforescape()

    squares = []
    for i in xrange(n_comps):
        singlecomponent = cv2.inRange(markers, i, i)

        contours, hierarchy = cv2.findContours(singlecomponent.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        norm = cv2.normalize(singlecomponent, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        print 'showing single component %d' % i
        cv2.imshow(wndname, norm)
        #waitforescape()

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
                cv2.imshow(wndname, singlecontour)
                #waitforescape()

                angle, topleft, size, square_bitmap = find_largest_overlapping_square(singlecontour, imagecutout)
                squares.append(Square(square_bitmap))

                x = topleft[0]
                y = topleft[1]
                frame = np.zeros(singlecontour.shape, np.uint8)
                cv2.rectangle(frame, (x,y), (x + size, y + size), 255, 1)
                cv2.imshow(wndname, frame)
                #waitforescape()

                rotation = cv2.getRotationMatrix2D( (frame.shape[0] / 2, frame.shape[1] / 2), angle, 1.0)
                rotatedframe = cv2.warpAffine(frame, rotation, frame.shape)
                cv2.imshow(wndname, rotatedframe)
                #waitforescape()

                print 'showing imagecutout'
                cv2.imshow(wndname, imagecutout)
                #waitforescape()

                rotatedframe = cv2.cvtColor( rotatedframe, cv2.COLOR_GRAY2BGR )
                cutout_with_frame = cv2.addWeighted(imagecutout, 1.0, rotatedframe, 0.5, 0)
                print 'showing imagecutout with frame'
                cv2.imshow(wndname, cutout_with_frame)
                waitforescape()


    return []


def find_largest_overlapping_square(singlecontour, imagecutout):

    overallMax = 0

    newcenter = (singlecontour.shape[0] / 2, singlecontour.shape[1] / 2)

    for angle in xrange(-16,17):
        rotation = cv2.getRotationMatrix2D(newcenter, angle, 1.0)
        rotatedcontour = cv2.warpAffine(singlecontour, rotation, singlecontour.shape)

        cv2.imshow(wndname, rotatedcontour)
        #waitforescape()

        offsetcontour = rotatedcontour.astype(np.float32) - 160

        for kernel_size in xrange(50,imagecutout.shape[0]):
            boxfiltered = cv2.boxFilter(offsetcontour, -1, (kernel_size, kernel_size), None, (0,0), False)

            norm = cv2.normalize(boxfiltered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imshow(wndname, norm)
            #waitforescape()


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
    if overallMax > 0:
        rotation = cv2.getRotationMatrix2D(newcenter, overallMaxAngle, 1.0)
        rotatedimagecutout = cv2.warpAffine(imagecutout, rotation, imagecutout.shape[0:2])
        bitmap = rotatedimagecutout[overallMaxLoc[0]:overallMaxLoc[0] + overallMaxKernelSize, overallMaxLoc[1]:overallMaxLoc[1] + overallMaxKernelSize]

        print 'showing largest overlapping square'
        cv2.imshow(wndname, bitmap)
        #waitforescape()

    return -overallMaxAngle, overallMaxLoc, overallMaxKernelSize, bitmap


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



