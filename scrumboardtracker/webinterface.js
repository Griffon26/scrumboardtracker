// Copyright 2016 Maurice van der Pot <griffon26@kfk4ever.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.




// The following debounce function has been taken from David Walsh's blog at
// https://davidwalsh.name/javascript-debounce-function and according to the footer
// was released under the MIT license: © David Walsh 2007-2016. All code MIT license.
// It appears to have been based on code from Underscore.js, which is also released
// under the MIT license.

// Returns a function, that, as long as it continues to be invoked, will not
// be triggered. The function will be called after it stops being called for
// N milliseconds. If `immediate` is passed, trigger the function on the
// leading edge, instead of the trailing.
function debounce(func, wait, immediate) {
    var timeout;
    return function() {
        var context = this, args = arguments;
        var later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        var callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
};

$(function() {
    var canvasInitialized = false;
    var canvas = new fabric.Canvas('canvas');


    var debounced = debounce(resizeCanvas, 250);
    window.addEventListener('resize', debounced, false);

    function resizeCanvas() {
        if(canvasInitialized) {

            availableWidth = window.innerWidth - 20;
            availableHeight = window.innerHeight - 20;

            if(availableWidth / availableHeight < canvas.backgroundImage.width / canvas.backgroundImage.height)
            {
                canvasWidth = availableWidth;
                canvasHeight = (canvasWidth * canvas.backgroundImage.height) / canvas.backgroundImage.width;
            }
            else
            {
                canvasHeight = availableHeight;
                canvasWidth = (canvasHeight * canvas.backgroundImage.width) / canvas.backgroundImage.height;
            }

            canvas.setWidth(canvasWidth);
            canvas.setHeight(canvasHeight);
            canvas.backgroundImage.scaleToWidth(canvasWidth);
            canvas.backgroundImage.scaleToHeight(canvasHeight);

            canvas.renderAll();
        }
    }

    fabric.Image.fromURL('/img', function(img) {

        if(!canvasInitialized)
        {
            canvasInitialized = true;

            //canvas.setDimensions({width:img.width, height:img.height});
            canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));

            resizeCanvas();

            // create a rectangle object
            var rect = new fabric.Rect({
                left: 100,
                top: 100,
                fill: 'red',
                width: 20,
                height: 20
            });

            canvas.observe("object:moving", function(e) {
                    var obj = e.target;

                    var halfw = obj.currentWidth/2;
                    var halfh = obj.currentHeight/2;
                    var bounds = {tl: {x: -halfw, y:-halfh},
                        br: {x: obj.canvas.width  - halfw, y: obj.canvas.height - halfh}
                    };

                    // top-left  corner

                    obj.top = Math.max(obj.top, bounds.tl.y)
                    obj.top = Math.min(obj.top, bounds.br.y)
                    obj.left = Math.max(obj.left, bounds.tl.x)
                    obj.left = Math.min(obj.left, bounds.br.x)
            });

            rect.lockMovementY = true
            rect.hasControls = false
             
            // "add" rectangle onto canvas
            canvas.add(rect);
        }
    });
});

