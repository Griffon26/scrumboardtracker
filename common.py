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

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

NOTE_SIZE = 50

# Determines the start and end of source and destination ranges for copying
# 'dest_size' pixels out of a range of 'source_size' pixels starting at offset
# 'offset'. The ranges given by this function can be used to copy a submatrix
# from a larger matrix without reading outside the source matrix or writing
# outside the destination matrix.
def determine_copy_ranges(offset, source_size, dest_size):

    dest_start = 0
    dest_end = dest_size

    src_start = offset
    src_end = offset + dest_size

    if src_start < 0:
        dest_start += -src_start
        src_start = 0
    if src_end > source_size:
        dest_end -= src_end - source_size
        src_end = source_size

    return src_start, src_end, dest_start, dest_end

def submatrix(bitmap_in, x, y, size):

    shape = [size, size]
    if len(bitmap_in.shape) == 3:
        shape.append(bitmap_in.shape[2])
    bitmap_out = np.zeros(shape, dtype=bitmap_in.dtype)

    src_row_start, src_row_end, dest_row_start, dest_row_end = determine_copy_ranges(y - size / 2, bitmap_in.shape[0], size)
    src_col_start, src_col_end, dest_col_start, dest_col_end = determine_copy_ranges(x - size / 2, bitmap_in.shape[1], size)

    bitmap_out[dest_row_start:dest_row_end,
               dest_col_start:dest_col_end] = bitmap_in[src_row_start:src_row_end,
                                                        src_col_start:src_col_end]
    return bitmap_out

def eucldistance(p1, p2):
    return cv2.norm(np.array(p1) - np.array(p2))

def correct_perspective(image, calibrationdata, fixedscale):
    aspectratio = float(calibrationdata['aspectratio'][0]) / float(calibrationdata['aspectratio'][1])

    if fixedscale:
        scale = 1
    else:
        scale = (1.0 * NOTE_SIZE) / calibrationdata['averagenotesize']
    #print 'scale: ', scale

    orderedcorners = np.array(calibrationdata['corners'], np.float32)
    originalwidth = eucldistance(orderedcorners[0], orderedcorners[1])
    scaledwidth = originalwidth * scale
    width = int(scaledwidth)
    height = int(width / aspectratio)

    #print 'width: ', width
    #print 'height: ', height

    correctedrectangle = np.array([(0,0), (width, 0), (width, height), (0, height)], np.float32)

    transformation = cv2.getPerspectiveTransform(orderedcorners, correctedrectangle)
    correctedimage = cv2.warpPerspective(image, transformation, (width, height))

    if calibrationdata['linepositions']:
        scaled_linepositions = [l * scale for l in calibrationdata['linepositions']]
    else:
        scaled_linepositions = None

    return correctedimage, scaled_linepositions

def remove_color_cast(image, calibrationdata):
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

def cvimage_to_qpixmap(image):
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, channel = image_rgb.shape
    qimage = QImage(image_rgb.data, width, height, width * 3, QImage.Format_RGB888)
    qpixmap = QPixmap(qimage)

    return qpixmap

class ImageLabel(QLabel):
    def __init__(self, image):
        QLabel.__init__(self, None)

        self.image = image
        self.redraw()

    def redraw(self):
        pixmap = cvimage_to_qpixmap(self.image)
        self.setGeometry(300, 300, pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)

class ImageDialog(QDialog):

    def __init__(self, images_and_text):
        super(ImageDialog, self).__init__()

        if not isinstance(images_and_text, (list, tuple)):
            images_and_text = [images_and_text]

        if not isinstance(images_and_text[0], (list, tuple)):
            images_and_text = [images_and_text]

        vbox = QVBoxLayout()

        for itemlist in images_and_text:
            hbox = QHBoxLayout()
            for item in itemlist:
                if isinstance(item, str):
                    hbox.addWidget(QLabel(item))
                else:
                    hbox.addWidget(ImageLabel(item))
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

def qimshow(images_and_text):
    dlg = ImageDialog(images_and_text)
    if dlg.exec_() != 1:
        raise Exception('Aborting. User pressed cancel.')


