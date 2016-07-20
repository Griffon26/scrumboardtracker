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
import os
import sys

from PyQt5 import QtWidgets

import readboard

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Update the scrumboard state using a webcam')
    parser.add_argument('-i', '--image', help='A photo of the scrumboard to process. ' +
                                              'If this is not specified an image is captured using a webcam.')
    parser.add_argument('-si', '--stateinput', type=argparse.FileType('r'),
                                               default=sys.stdin,
                                               help='The JSON file containing the initial scrumboard state. ' +
                                                    'If this is not specified the original state is taken from stdin.')
    parser.add_argument('-so', '--stateoutput', type=argparse.FileType('w'),
                                                default=sys.stdout,
                                                help='The file that the scrumboard state should be written to. ' +
                                                     'If this is not specified the updated state is written to stdout.')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug prints and dialogs')
    args = parser.parse_args()

    if args.debug:
        app = QtWidgets.QApplication(sys.argv)

    old_state = args.stateinput.read()
    new_state = readboard.readboard(old_state, imagefile=args.image, debug=args.debug)
    args.stateoutput.write(new_state + '\n')

