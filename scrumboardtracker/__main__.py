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

import cv2
from datetime import datetime
import json
import os
from pprint import pprint
import sys
from twisted.internet import error, protocol, reactor

import board
from boardreader import readboard

# To make opencv2 compatible with the opencv3 API we use
if cv2.__version__.startswith('2'):
    cv2.LINE_AA = cv2.CV_AA

class ProcessIOProtocol(protocol.ProcessProtocol):

    def __init__(self, stdin, process_success_cb, process_failure_cb, print_stderr=False):
        self.stdin = stdin
        self.stdout = ''
        self.stderr = ''
        self.status = None
        self.print_stderr = print_stderr

        self.process_success_cb = process_success_cb
        self.process_failure_cb = process_failure_cb

    def connectionMade(self):
        self.transport.write(self.stdin)
        self.transport.closeStdin()

    def outReceived(self, data):
        self.stdout += data

    def errReceived(self, data):
        self.stderr += data
        if self.print_stderr:
            sys.stderr.write(data)

    def processEnded(self, reason):
        if isinstance(reason.value, error.ProcessDone):
            self.process_success_cb(self.stdout, self.stderr)
        else:
            self.process_failure_cb(self.stdout, self.stderr)

class ScrumBoardTracker():

    def __init__(self):
        self.scrumboardfile = 'scrumboardstate.json'
        self.scrumboard = board.Scrumboard()

    def schedule_timer(self):
        print 'Scheduling next board update in 10 seconds...\n'
        reactor.callLater(10, self.startBoardReader)

    def startBoardReader(self):
        print 'Starting board update...'

        boardstate = json.dumps(self.scrumboard.to_serializable())
        brp = ProcessIOProtocol(boardstate, self.handleProcessOutput, self.handleProcessFailure, print_stderr=True)
        reactor.spawnProcess(brp, sys.executable, [sys.executable, '-m', 'scrumboardtracker.boardreader'])

    def handleProcessOutput(self, stdout, stderr):
        boardstate = json.loads(stdout)

        # Write various data for debugging to timestamped files in the logs directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        with open('logs/%s_1_initial_board_state.json' % timestamp, 'wb') as f:
            json.dump(self.scrumboard.to_serializable(), f)

        updated_scrumboard = board.Scrumboard()
        updated_scrumboard.from_serializable(boardstate)

        differences = updated_scrumboard.diff(self.scrumboard)
        self.scrumboard = updated_scrumboard


        with open('logs/%s_2_differences.txt' % timestamp, 'wb') as f:
            pprint(differences, f)

        cv2.imwrite('logs/%s_3_board.png' % timestamp, self.scrumboard.bitmap)

        annotated_bitmap = self.scrumboard.bitmap.copy()
        for note in self.scrumboard.tasknotes:
            text_size, baseline = cv2.getTextSize(str(note.taskid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x = note.position[0] - text_size[0] / 2
            y = note.position[1] + text_size[1] / 2
            cv2.putText(annotated_bitmap, str(note.taskid), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite('logs/%s_4_annotated_board.png' % timestamp, annotated_bitmap)

        # Before saving board state to json, remove the bitmap because it is
        # very large and not used when read from file
        boardstate['bitmap'] = []
        with open(self.scrumboardfile, 'wb') as f1, \
             open('logs/%s_5_final_board_state.json' % timestamp, 'wb') as f2:
            json.dump(boardstate, f1)
            json.dump(boardstate, f2)

        print 'Board state updated (see logs/%s_*)' % timestamp

        self.schedule_timer()

    def handleProcessFailure(self, stdout, stderr):
        print 'Board update process failed: stderr output was %s' % stderr

        self.schedule_timer()

    def run(self):

        try:
            os.mkdir('logs')
        except OSError:
            pass

        if os.path.exists(self.scrumboardfile):
            with open(self.scrumboardfile, 'rb') as f:
                boardstate = f.read()

            if len(boardstate) > 0:
                self.scrumboard.from_serializable(json.loads(boardstate))

        self.startBoardReader()
        reactor.run()

def main():
    tracker = ScrumBoardTracker()
    tracker.run()


if __name__ == '__main__':
    main()
