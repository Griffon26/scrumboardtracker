# Copyright 2014 Maurice van der Pot <griffon26@kfk4ever.com>
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
import os
import subprocess as sp

def get_this_filename():
    import inspect
    stack = inspect.stack()
    _, filename, _, _, _, _ = stack[0]
    return filename

def grab():
    capturedir = os.path.expanduser('~/.scrumboardtracker')
    capturefile = capturedir + '/capture.jpg'

    try:
        os.mkdir(capturedir)
    except OSError:
        pass

    this_location = os.path.dirname(get_this_filename())

    sp.call('%s/capture.sh %s' % (this_location, capturefile), shell=True)
    image = cv2.imread(capturefile, 1);
    os.remove(capturefile)
    return image
