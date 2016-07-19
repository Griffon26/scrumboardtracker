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

import readboard

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Update the scrumboard state using a webcam')
    parser.add_argument('filename', nargs='?', help='the JSON file containing the scrumboard state to be updated. ' +
                                                    'If this is not specified the original state is taken from stdin ' +
                                                    'and the updated state is written to stdout.')
    args = parser.parse_args()

    if args.filename:
        if os.path.exists(args.filename):
            with open(args.filename, 'rb') as f:
                old_state = f.read()
        else:
                old_state = ''
    else:
        old_state = sys.stdin.read()

    new_state = readboard.readboard(old_state)

    if args.filename:
        with open(args.filename, 'wb') as f:
            f.write(new_state)
    else:
        sys.stdout.write(new_state + '\n')

