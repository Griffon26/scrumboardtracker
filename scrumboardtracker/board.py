# Copyright 2014-2016 Maurice van der Pot <griffon26@kfk4ever.com>
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

import numpy as np
import itertools

class Square():
    def __init__(self, bitmap, position):
        self.bitmap = bitmap
        self.position = position

    def lookslike(self, otherthingwithbitmap):
        #diff = cv2.norm(self.bitmap, otherthingwithbitmap.bitmap, cv2.NORM_L1)
        #print 'lookslike calculated diff of ', diff
        # self.bitmap ~ otherthingwithbitmap.bitmap
        return False

class TaskNote():
    def __init__(self, taskid, bitmap, position, state):
        self.taskid = taskid
        self.bitmap = bitmap
        self.position = position
        self.state = state

    def to_serializable(self):
        return { 'bitmap' : self.bitmap.tolist(),
                 'position' : self.position,
                 'state' : self.state }

    def from_serializable(self, data):
        self.bitmap = np.array(data['bitmap'], dtype=np.uint8)
        self.state = data['state']
        self.position = data['position']

    def _setstate(self, newposition, newstate):
        self.position = newposition
        self.state = newstate

class Scrumboard():
    def __init__(self, linepositions=None):
        self.linepositions = linepositions
        self.bitmap = np.array([])
        self.tasknotes = []
        self.states = ['todo', 'busy', 'blocked', 'in review', 'done']

    def to_serializable(self):
        return { 'bitmap' : self.bitmap.tolist(),
                 'tasknotes' : [tasknote.to_serializable() for tasknote in self.tasknotes] }

    def from_serializable(self, data):
        self.bitmap = np.array(data['bitmap'], dtype=np.uint8)
        self.tasknotes = []
        for tasknote_data in data['tasknotes']:
            tasknote = TaskNote(len(self.tasknotes), None, None, None)
            tasknote.from_serializable(tasknote_data)
            self.tasknotes.append(tasknote)

    def get_state_from_position(self, position):
        for i, linepos in enumerate(self.linepositions):
            if position[0] < linepos:
                return self.states[i]

        return self.states[-1]

    def get_position_from_state(self, state):
        positions = [0] + self.linepositions + [None]
        stateindex = self.states.index(state)
        return positions[stateindex], positions[stateindex + 1]

    def set_bitmap(self, image):
        self.bitmap = image

    def add_tasknote(self, square):
        state = self.get_state_from_position(square.position)
        tasknote = TaskNote(len(self.tasknotes), square.bitmap, square.position, state)
        self.tasknotes.append(tasknote)
        return tasknote

    def move_tasknote(self, tasknote, newposition):
        newstate = self.get_state_from_position(newposition)
        self.tasknotes[tasknote.taskid]._setstate(newposition, newstate)
        return newstate

    def diff(self, otherboard):
        differences = {}
        for i, states in enumerate(itertools.izip_longest((note.state for note in otherboard.tasknotes),
                                                          (note.state for note in self.tasknotes))):
            if states[0] != states[1]:
                differences[i] = states
        return differences
