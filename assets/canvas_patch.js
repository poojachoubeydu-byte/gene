/**
 * canvas_patch.js
 * ---------------
 * Plotly's WebGL (Scattergl) traces call getImageData() on 2-D canvas
 * contexts without the willReadFrequently hint, which triggers a Chrome
 * performance warning.  This patch intercepts every getContext('2d') call
 * and injects the attribute so the browser can optimise the readback path.
 *
 * Loaded automatically by Dash from the assets/ directory at startup.
 */
(function () {
    'use strict';
    var _orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, attrs) {
        if (type === '2d') {
            attrs = Object.assign({ willReadFrequently: true }, attrs || {});
        }
        return _orig.call(this, type, attrs);
    };
})();
