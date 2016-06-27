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

    def __init__(self, images, text=None):
        super(ImageDialog, self).__init__()

        if not isinstance(images, (list, tuple)):
            images = [images]

        if not isinstance(images[0], (list, tuple)):
            images = [images]

        vbox = QVBoxLayout()

        if text != None:
            vbox.addWidget(QLabel(text))

        for imagelist in images:
            hbox = QHBoxLayout()
            for image in imagelist:
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


