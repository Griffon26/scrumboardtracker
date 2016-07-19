#!/usr/bin/env python2

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

import argparse
import cv2
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonfile', help='the JSON file containing the scrumboard state from which to extract a task note bitmap')
    parser.add_argument('bitmapfile', help='the name of the file to save the bitmap to')
    parser.add_argument('taskid', type=int, nargs='?', help='the taskid of the task note to be extracted. If this is not provided the user will be asked to enter one.')
    args = parser.parse_args()

    with open(args.jsonfile, 'rb') as f:
        boardstate = json.load(f)

    tasknotes = boardstate['tasknotes']

    if args.taskid == None:
        selectedtaskid = None
        while selectedtaskid == None:
            try:
                userinput = raw_input("This json file contains %d task notes.\n" % len(tasknotes) +
                                      "Please enter the ID of the task whose bitmap must be saved to file (0 - %d) " % (len(tasknotes) - 1))
                selectedtaskid = int(userinput)
            except ValueError:
                pass
    else:
        selectedtaskid = args.taskid

    cv2.imwrite(args.bitmapfile, np.array(tasknotes[selectedtaskid]['bitmap']))
    print 'Task note bitmap saved to %s' % args.bitmapfile
