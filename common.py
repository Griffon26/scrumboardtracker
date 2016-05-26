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

import cv2
import numpy as np

NOTE_SIZE = 50

def eucldistance(p1, p2):
    return cv2.norm(np.array(p1) - np.array(p2))

def correct_perspective(image, calibrationdata, fixedscale):
    aspectratio = float(calibrationdata['aspectratio'][0]) / float(calibrationdata['aspectratio'][1])

    if fixedscale:
        scale = 1
    else:
        scale = (1.0 * NOTE_SIZE) / calibrationdata['averagenotesize']
    print('scale', scale)

    orderedcorners = np.array(calibrationdata['corners'], np.float32)
    originalwidth = eucldistance(orderedcorners[0], orderedcorners[1])
    scaledwidth = originalwidth * scale
    width = int(scaledwidth)
    height = int(width / aspectratio)

    print 'width: ', width
    print 'height: ', height

    correctedrectangle = np.array([(0,0), (width, 0), (width, height), (0, height)], np.float32)

    transformation = cv2.getPerspectiveTransform(orderedcorners, correctedrectangle)
    correctedimage = cv2.warpPerspective(image, transformation, (width, height))

    return correctedimage
