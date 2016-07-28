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

import sys

from twisted.internet import error, protocol

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


