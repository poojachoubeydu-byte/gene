// Prevent Plotly from fetching KaTeX from external CDN.
// Plotly checks for window.katex before making the CDN request;
// providing a minimal stub here satisfies that check without any network call.
(function () {
    if (!window.katex) {
        window.katex = {
            renderToString: function (s) { return s; },
            render: function () {}
        };
    }
})();
