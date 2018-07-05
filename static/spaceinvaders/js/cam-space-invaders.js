//import * as tf from '@tensorflow/tfjs';
// A webcam class that generates Tensors from the images from the webcam.
//const webcam = new Webcam(document.getElementById('webcam'));
//@TODO: add example count for each button

let video;
var stream, imageCapture;
let dataForFetch = {
    4: [], 6: []
};

const tsParams = {
    units: 100,
    epochs: 20,
    batchSize: 0.4,
    learningRate: 0.0001,
    threshold: 0.95
}

// watch form changes
document.querySelectorAll('select').forEach(select => {
    select.onchange = function() {
        const value = parseFloat(this.value);
        switch (this.id) {
            case 'inpLearningRate':
                tsParams.learningRate = value;
                break;
            case 'inpBatchSize':
                tsParams.batchSize = value;
                break;
            case 'inpEpochs':
                tsParams.epochs = value;
                break;
            case 'inpHiddenUnits':
                tsParams.units = value;
                break;
        }
    }
})

document.querySelector('#inferenceThreshold').onchange = function() {
    const value = parseFloat(this.value);
    document.querySelector('#thresholdMonitor').textContent = value + '%';
    tsParams.threshold = value / 100;
}

const gameHash = Math.abs(hashCode(`${new Date().getTime()  * 1000} in-game`));
let pendingForSend = [];

function getMediaStream() {
    window.navigator.mediaDevices.getUserMedia({video: true})
        .then(function (mediaStream) {
            stream = mediaStream;
            let mediaStreamTrack = mediaStream.getVideoTracks()[0];
            this.imageCapture = new ImageCapture(mediaStreamTrack);
        })
        .catch(error);

}

if (window.sendInterval) {
    // avoid in case of duplicate javascript loading
    clearInterval(window.sendInterval);
}

window.sendInterval = setInterval( () => {
    if (pendingForSend.length) {
        console.log(`Sending ${pendingForSend.length} images...`);
        const formData = new FormData();
        pendingForSend.forEach((dataItem) => {
            const playerName = 'garbage'
            const {blob, number, date, sampleNumber} = dataItem;
            const imageFileName = `${gameHash}-${playerName}-${date}-${number}-${sampleNumber}.jpg`;
            formData.set(imageFileName, blob);
            fetchData(formData);
        });
        pendingForSend.length = 0;
    }
}, 3000);

function takePhoto(img, label) {
    const batchedImage = takeCapture();
    const tsLabel = label === 4 ? 0 : 1;
    modelOperator.addExample(batchedImage, tsLabel);
    imageCapture.takePhoto()
        .then(blob => {
            var
                ctx = img.getContext('2d')
                , img1 = new Image
                , totalElement = img.closest('.thumb-box').querySelector('.total')
                , url = window.URL
                , imageDataList = dataForFetch[label];

            img1.src = url.createObjectURL(blob);
            const dataObject = {
                blob,
                number: label,
                date: new Date().getTime(),
                sampleNumber: imageDataList.length
            }
            imageDataList.push(dataObject);
            pendingForSend.push(dataObject);
            totalElement.innerHTML = imageDataList.length;
            img1.onload = function () {
                var canvas = img;
                ctx.drawImage(img1, 0, 0, img1.width, img1.height, 0, 0, canvas.width, canvas.height);
                url.revokeObjectURL(blob);
            }


        })
        .catch(error);
};

function sleep(milliseconds) {
    var start = new Date().getTime();
    for (var i = 0; i < 1e7; i++) {
        if ((new Date().getTime() - start) > milliseconds) {
            break;
        }
    }
}

function error(error) {
    console.error('error:', error);
}

function handleVideo(stream) {
    video.src = window.URL.createObjectURL(stream);
}

function videoError(e) {
    // do something
}

function stringClean(str) {
    return str.replace(/[ -]/g, '_').toLocaleLowerCase();
}

function isDataForFetchEmpty() {
    for (let data in dataForFetch) {
        if (dataForFetch[data].length === 0) {
            return true
        }
    }
    return false;
}

function hashCode(str) {
    let
        i = 0
        , hash = 0
        , character;
    for (i; i < str.length; i++) {
        character = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + character;
        hash = hash & hash;
    }
    return hash;
}

function fetchData(data) {
    const
        url = '/upload';
    let
        headers = new Headers();
    // myHeaders.append('Content-Type', 'multipart/form-data');
    headers.append('Accept', 'application/json');

    const myInit = {
        method: 'POST'
        , mode: 'cors'
        , headers: headers
        , body: data
    };
    fetch(url, myInit)
}

function redirect(url) {
    return window.location.href = url;
}



function init() {
    video = document.querySelector("#webcam");

    document.querySelectorAll('.record-button').forEach((elem, index, array) => elem.addEventListener("click", (e) => {
        let
            canvas = e.target.parentElement.getElementsByTagName('canvas')[0]
            , imageNum = +e.target.value;
        takePhoto(canvas, imageNum);
    }, false));

    document.querySelector('#train').onclick = function() {
        syncStarted = false;
        modelOperator.train();
    };


    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.oGetUserMedia;

    if (navigator.getUserMedia) {
        navigator.getUserMedia({video: true}, handleVideo, videoError)
    }

    getMediaStream();
}

var syncStarted = false;

async function startTensorSync() {
    if (syncStarted || !modelOperator.mobilenet) return;
    const webcamElement = document.querySelector('#webcam');
    if (!webcamElement) {
        alert('Webcam not found');
        return;
    }
    syncStarted = true;
    while (syncStarted) {
        await syncTensor(webcamElement);
        await delay(30);
    }
}

const delay = (time = 300) => {
    return new Promise((resolve) => {
        setTimeout(resolve, time)
    })
}

let hiddenCanvas = document.createElement('canvas');

// canvas debug
// document.body.appendChild(hiddenCanvas);
hiddenCanvas.setAttribute('width', '224');
hiddenCanvas.setAttribute('height', '224');
hiddenCanvas.style.display = 'inline-block';
hiddenCanvas.style.width = '224px';
hiddenCanvas.style.height = '224px';
hiddenCanvas.style.position = 'absolute';
// canvas debug end


let hiddenCanvasCtx = hiddenCanvas.getContext('2d');

/**
    Take picture using canvas. tf.fromPixels is expecting canvas object.
**/
function takeCapture() {
    hiddenCanvasCtx.drawImage(video, 0, 0, 224, 224);
    const fromPixels = tf.fromPixels(hiddenCanvas);
    const croppedImage = cropImage(fromPixels);
    const batchedImage = croppedImage.expandDims(0);
    // Convert pixel colors from 0 to 255 to (-1) to (+1)
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}

async function syncTensor() {
    // Take image from camera
    let img;
    tf.tidy( () => {
        img = takeCapture();
        return img;
    });

    // 37 left
    // 39 right
    if (modelOperator.model && !modelOperator.isTraining) {
        // Update UI
        // First, stop the ship and clean previous selection
        game.keyUp(39);
        game.keyUp(37);
        const btnLeft = document.querySelector('button#left');
        const btnRight = document.querySelector('button#right');
        btnLeft.style.border = null;
        btnRight.style.border = null;

        // Second, Predict using modelOperator.mobilenet and then the custom model and
        // check whether probability is higher than the threshold
        const predictedClass = modelOperator.mobilenetPredict(img)
        const probability = await predictedClass.data();
        const pos = await predictedClass.argMax().data();
        predictedClass.dispose();
        const max = Math.max(...probability);
        const maxMonitor = document.querySelector('#probabilityMonitor');
        if (max <= tsParams.threshold) {
            // Below threshold. Paint monitor number as red and quit.
            maxMonitor.textContent = `💩 ${max.toFixed(3)}`;
            maxMonitor.style.color = 'red';
            return;
        }
        // Shift ship direction and mark border according to prediction
        maxMonitor.textContent = `👍 ${max.toFixed(3)}`;
        maxMonitor.style.color = 'green';
        const borderStyle = '2px solid yellow'
        switch (pos[0]) {
            case 0:
                game.moveToTheLeft();
                btnLeft.style.border = borderStyle;
                break;
            case 1:
                game.moveToTheRight();
                btnRight.style.border = borderStyle;
                break;
        }
        await tf.nextFrame();
    }
}

function cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}


function updateTrainingStatus(value) {
    document.querySelector('#train-status').textContent = value;
}

window.modelOperator = new ModelOperator(2);
modelOperator.updateTrainingStatus = updateTrainingStatus;

/* just call */
document.addEventListener("DOMContentLoaded", init);