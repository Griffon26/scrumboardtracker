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

from boardreader import readboard

def main():
    old_state = '{}'
    new_state = readboard(old_state)
    print new_state

if __name__ == '__main__':
    main()
