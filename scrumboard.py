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

wndname = "Scrumboard"


def waitforescape():
    while 1:
        k = cv2.waitKey(1) & 0xFF;
        if k == 27:
            break

if __name__ == "__main__":
    cv2.namedWindow( wndname, 1 );
    cv2.moveWindow(wndname, 100, 100);

    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))

    image = cv2.imread("webcam.jpg", 1);
    cv2.imshow(wndname, image)
    waitforescape()

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
    waitforescape()

    hsv = cv2.cvtColor( correctedimage, cv2.COLOR_BGR2HSV )

    _, saturation, _ = cv2.split(hsv)
    print "Showing saturation of corrected image"
    cv2.imshow(wndname, saturation)
    waitforescape()
