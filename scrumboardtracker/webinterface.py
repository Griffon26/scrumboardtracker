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
import json
import math
import os

from nevow import tags as T, stan
from nevow import appserver, loaders, rend, static
from nevow import inevow
from twisted.internet import error, protocol, reactor
from twisted.web import resource, server

from boardreader import imagefuncs

def load_calibrationdata():
    with open('calibrationdata.json', 'rb') as f:
        calibrationdata = json.loads('\n'.join(f.readlines()))
    return calibrationdata

def save_calibrationdata(calibrationdata):
    with open('calibrationdata.json', 'wb') as f:
        json.dump(calibrationdata, f)

class ImageServer(resource.Resource):
    def __init__(self, webcam):
        self.webcam = webcam

    def render(self, ctx):
        d = self.webcam.capture()
        d.addCallback(self.captureDone, ctx)
        return server.NOT_DONE_YET

    def captureDone(self, img, ctx):
        retval, imgdata = cv2.imencode('.png', img)
        data = imgdata.flatten().tostring()

        request = inevow.IRequest(ctx)
        request.setHeader("content-type", 'image/png')
        request.setHeader("content-length", str(len(data)))
        request.write(data)
        request.finish()

class TransformedImageServer(resource.Resource):
    def __init__(self, webcam, calibrationdata):
        self.webcam = webcam
        self.calibrationdata = calibrationdata

    def render(self, ctx):
        d = self.webcam.capture()
        d.addCallback(self.captureDone, ctx)
        return server.NOT_DONE_YET

    def captureDone(self, img, ctx):
        correctedimage = imagefuncs.correct_perspective(imagefuncs.remove_color_cast(img, self.calibrationdata), self.calibrationdata, True)

        retval, imgdata = cv2.imencode('.png', correctedimage)
        data = imgdata.flatten().tostring()

        request = inevow.IRequest(ctx)
        request.setHeader("content-type", 'image/png')
        request.setHeader("content-length", str(len(data)))
        request.write(data)
        request.finish()

class MainPage(rend.Page):

    docFactory = loaders.stan(
        T.html[
            T.head[
                T.title(id='title', name='main')["Griffon26's nevow test"],
            ],
            T.body[
                T.a(href='calibration1.html')['Start calibration'],
                T.div(render=T.directive('postdata'))
            ]
        ]
    )

    def render_postdata(self, context, data):
        linepositions_json = context.arg('linepositions')
        if linepositions_json:
            self.calibrationdata['linepositions'] = json.loads(linepositions_json)
            save_calibrationdata(self.calibrationdata)
        return ''

    def __init__(self, calibrationdata):
        rend.Page.__init__(self)
        self.calibrationdata = calibrationdata

class CalibrationPage1(rend.Page):

    docFactory = loaders.stan(
        T.html[
            T.head[
                T.title(id='title', name='calibration1')["Griffon26's nevow test - calibration part 1"],
                T.script(src="http://cdnjs.cloudflare.com/ajax/libs/fabric.js/1.6.3/fabric.min.js"),
                T.script(src="http://code.jquery.com/jquery-2.1.0.min.js"),
                T.script(src="webinterface.js")
            ],
            T.body[
                T.script(render=T.directive('calibrationdata')),
                stan.Tag('canvas')(id='canvas', style="border:1px solid #000000;"),
                T.form(action='calibration2.html', method='post', id='form', onSubmit='return setFormFieldsCalib1()')[
                    T.input(type='number', name='aspectx', step='any'),
                    T.input(type='number', name='aspecty', step='any'),
                    T.input(type='hidden', name='corners'),
                    T.input(type='submit', value='Next')
                ],
            ]
        ]
    )

    def render_calibrationdata(self, context, data):
        return T.script['var calibrationdata = %s' % json.dumps(self.calibrationdata)]

    def __init__(self, calibrationdata):
        rend.Page.__init__(self)
        self.calibrationdata = calibrationdata

class CalibrationPage2(rend.Page):

    docFactory = loaders.stan(
        T.html[
            T.head[
                T.title(id='title', name='calibration2')["Griffon26's nevow test - calibration part 2"],
                T.script(src="http://cdnjs.cloudflare.com/ajax/libs/fabric.js/1.6.3/fabric.min.js"),
                T.script(src="http://code.jquery.com/jquery-2.1.0.min.js"),
                T.script(src="webinterface.js")
            ],
            T.body[
                T.script(render=T.directive('calibrationdata')),
                stan.Tag('canvas')(id='canvas', style="border:1px solid #000000;"),
                T.form(action='/', method='post', id='form', onSubmit='return setFormFieldsCalib2()')[
                    T.input(type='hidden', name='linepositions'),
                    T.input(type='submit', value='Next')
                ],
            ]
        ]
    )

    def sort_corners_clockwise(self, corners):
        #
        # Sort the corners such that the top left corner is the first in the list and the corners are in clockwise order
        #

        vertically_sorted_corners_with_index = sorted(enumerate(corners), key=lambda t: t[1][1])
        topindex, toppoint = vertically_sorted_corners_with_index[0]

        rightindex = (topindex + 1) % 4
        rightpoint = corners[rightindex]
        leftindex = (topindex + 4 - 1) % 4
        leftpoint = corners[leftindex]

        def abs_distance_per_axis(p1, p2):
            return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

        xdiff_right, ydiff_right = abs_distance_per_axis(toppoint, rightpoint)
        xdiff_left, ydiff_left = abs_distance_per_axis(toppoint, leftpoint)

        acos_left = math.acos(abs(xdiff_left) / imagefuncs.eucldistance(leftpoint, toppoint))
        acos_right = math.acos(abs(xdiff_right) / imagefuncs.eucldistance(rightpoint, toppoint))

        if acos_left < acos_right:
            othertopindex = leftindex
            othertoppoint = leftpoint
        else:
            othertopindex = rightindex
            othertoppoint = rightpoint

        if toppoint[0] < othertoppoint[0]:
            startingpointindex = topindex
            direction = othertopindex - topindex
        else:
            startingpointindex = othertopindex
            direction = topindex - othertopindex

        if direction > 0:
            corners = (corners * 2)[startingpointindex:startingpointindex + 4]
        else:
            corners = (corners * 2)[startingpointindex + 1:startingpointindex + 4 + 1]
            corners.reverse()

        return corners

    def render_calibrationdata(self, context, data):
        corners = json.loads(context.arg('corners'))
        corners = self.sort_corners_clockwise(corners)
        self.calibrationdata['corners'] = corners

        self.calibrationdata['aspectratio'] = [ float(context.arg('aspectx')), float(context.arg('aspecty')) ]

        save_calibrationdata(self.calibrationdata)

        return T.script['var calibrationdata = %s' % json.dumps(self.calibrationdata)]

    def __init__(self, calibrationdata):
        rend.Page.__init__(self)
        self.calibrationdata = calibrationdata


class WebInterface:

    def __init__(self, webcam):
        calibrationdata = load_calibrationdata()

        imageServer = ImageServer(webcam)
        transformedImageServer = TransformedImageServer(webcam, calibrationdata)

        root = MainPage(calibrationdata)
        root.putChild('calibration1.html', CalibrationPage1(calibrationdata))
        root.putChild('calibration2.html', CalibrationPage2(calibrationdata))
        root.putChild('getImage.cgi', imageServer)
        root.putChild('getTransformedImage.cgi', transformedImageServer)
        root.putChild('webinterface.js', static.File(os.path.dirname(__file__) + '/webinterface.js'))

        self.site = appserver.NevowSite( root )

    def start(self):
        reactor.listenTCP(8080, self.site)

