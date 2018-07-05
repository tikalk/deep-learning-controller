
function whenAvailable(name, callback) {
    console.log("when available?")
    var interval = 10; // ms
    window.setTimeout(function() {
        if (window[name]) {
            console.log("when available? found!!")
            callback(window[name]);
        } else {
            window.setTimeout(arguments.callee, interval);
        }
    }, interval);
}