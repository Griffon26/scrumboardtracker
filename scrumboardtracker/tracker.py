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
from datetime import datetime
import json
import os
from pprint import pprint
import sys
from twisted.internet import reactor

import board
from boardreader import imagefuncs, readboard
from processioprotocol import ProcessIOProtocol

# To make opencv2 compatible with the opencv3 API we use
if cv2.__version__.startswith('2'):
    cv2.LINE_AA = cv2.CV_AA

class ScrumBoardTracker():

    def __init__(self, webcam):
        self.scrumboardfile = 'scrumboardstate.json'
        self.scrumboard = board.Scrumboard()
        self.webcam = webcam

    def schedule_timer(self):
        print 'Scheduling next board update in 10 seconds...\n'
        reactor.callLater(10, self.startImageCapture)

    def startImageCapture(self):
        deferred = self.webcam.capture_filtered()
        deferred.addCallback(self.startBoardUpdate)

    def startBoardUpdate(self, image):
        print 'Starting board update...'

        with open('calibrationdata.json', 'rb') as f:
            calibrationdata = json.loads('\n'.join(f.readlines()))

        colorfiximage = imagefuncs.remove_color_cast(image, calibrationdata)
        correctedimage = imagefuncs.correct_perspective(colorfiximage, calibrationdata, False)

        self.scrumboard.set_bitmap(correctedimage)

        boardstate = json.dumps(self.scrumboard.to_serializable())
        brp = ProcessIOProtocol(boardstate, self.handleProcessOutput, self.handleProcessFailure, print_stderr=True)
        reactor.spawnProcess(brp, sys.executable, [sys.executable, '-d', '-m', 'scrumboardtracker.boardreader'])

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

    def start(self):

        try:
            os.mkdir('logs')
        except OSError:
            pass

        if os.path.exists(self.scrumboardfile):
            with open(self.scrumboardfile, 'rb') as f:
                boardstate = f.read()

            if len(boardstate) > 0:
                self.scrumboard.from_serializable(json.loads(boardstate))

        self.startImageCapture()

