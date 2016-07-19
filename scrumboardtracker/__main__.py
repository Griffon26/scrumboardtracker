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
        print 'Scheduling next board update in 10 seconds'
        reactor.callLater(10, self.startBoardReader)

    def startBoardReader(self):
        print 'Starting board update...'

        boardstate = json.dumps(self.scrumboard.to_serializable())
        brp = ProcessIOProtocol(boardstate, self.handleProcessOutput, self.handleProcessFailure, print_stderr=True)
        reactor.spawnProcess(brp, sys.executable, [sys.executable, '-m', 'scrumboardtracker.boardreader'])

    def handleProcessOutput(self, stdout, stderr):
        boardstate = stdout

        with open(self.scrumboardfile, 'wb') as f:
            f.write(boardstate)

        updated_scrumboard = board.Scrumboard()
        updated_scrumboard.from_serializable(json.loads(boardstate))

        differences = updated_scrumboard.diff(self.scrumboard)
        self.scrumboard = updated_scrumboard


        # Write board bitmap and differences to timestamped files in logs directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        with open('logs/%s_1_differences.txt' % timestamp, 'wb') as f:
            pprint(differences, f)
        cv2.imwrite('logs/%s_2_board.png' % timestamp, self.scrumboard.bitmap)

        print 'Board state updated, differences:', differences

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
