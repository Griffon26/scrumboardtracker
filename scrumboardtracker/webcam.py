# Copyright 2016 Maurice van der Pot <griffon26@kfk4ever.com>
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
import os
from twisted.internet import defer, reactor

from processioprotocol import ProcessIOProtocol

def get_this_filename():
    import inspect
    stack = inspect.stack()
    _, filename, _, _, _, _ = stack[0]
    return filename

class WebCam:
    def __init__(self):
        capturedir = os.path.expanduser('~/.scrumboardtracker')
        try:
            os.mkdir(capturedir)
        except OSError:
            pass
        self.capturefile = capturedir + '/capture.jpg'

        self.capture_running = False

        self.capture_deferred = None

        self.required_nr_of_images = 3
        self.capture_sequence = []
        self.capture_filtered_deferred = None

        self.last_capture_size = (0,0)

    def _handleCaptureSuccess(self, stdout, stderr):
        image = cv2.imread(self.capturefile, 1);
        os.remove(self.capturefile)

        self.last_capture_size = (image.shape[1], image.shape[0])

        if self.capture_deferred:
            self.capture_deferred.callback(image)
            self.capture_deferred = None

        if self.capture_filtered_deferred:
            self.capture_sequence.append(image)

            if len(self.capture_sequence) == self.required_nr_of_images:
                image = np.zeros(self.capture_sequence[0].shape, self.capture_sequence[0].dtype)
                for i in range(self.required_nr_of_images):
                    image = cv2.addWeighted(image, 1, self.capture_sequence[i], 1.0 / self.required_nr_of_images, 0)
                self.capture_filtered_deferred.callback(image)
                self.capture_filtered_deferred = None
            else:
                self._startCapture()

    def _handleCaptureFailure(self, stdout, stderr):
        print 'fail, stderr = %s' % stderr

    def _startCapture(self):
        brp = ProcessIOProtocol('', self._handleCaptureSuccess, self._handleCaptureFailure, print_stderr=True)
        this_location = os.path.dirname(get_this_filename())
        executable = '%s/capture.sh' % this_location
        reactor.spawnProcess(brp, executable, [executable, self.capturefile])

    def capture(self):
        d = defer.Deferred()
        if not self.capture_running:
            self._startCapture()

        self.capture_deferred = d
        return d

    def capture_filtered(self):
        d = defer.Deferred()
        if not self.capture_running:
            self._startCapture()

        self.capture_sequence = []
        self.capture_filtered_deferred = d
        return d

    def get_last_capture_size(self):
        return self.last_capture_size
