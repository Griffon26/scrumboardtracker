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
import os

from nevow import tags as T, stan
from nevow import appserver, loaders, rend, static
from nevow import inevow
from twisted.internet import error, protocol, reactor
from twisted.web import resource, server

class MainPage(rend.Page):

    docFactory = loaders.stan(
        T.html[
            T.head[
                T.title["Griffon26's nevow test"],
                T.script(src="http://cdnjs.cloudflare.com/ajax/libs/fabric.js/1.4.0/fabric.min.js"),
                T.script(src="http://code.jquery.com/jquery-2.1.0.min.js"),
                T.script(src="webinterface.js")
            ],
            T.body[
                stan.Tag('canvas')(id='canvas', style="border:1px solid #000000;"),
            ]
        ]
    )

    def __init__(self, webcam):
        rend.Page.__init__(self)
        self.webcam = webcam

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

class WebInterface:

    def __init__(self, webcam):
        imgserver = ImageServer(webcam)

        root = MainPage(webcam)
        root.putChild('img', imgserver)
        root.putChild('webinterface.js', static.File(os.path.dirname(__file__) + '/webinterface.js'))

        self.site = appserver.NevowSite( root )

    def start(self):
        reactor.listenTCP(8080, self.site)

