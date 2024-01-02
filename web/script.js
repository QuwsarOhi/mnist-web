const CANVAS_SIZE = 280;
var canvas, ctx, flag = false;
var prevX = 0;
var currX = 0;
var prevY = 0;
var currY = 0;
var dot_flag = false;
var x = "black", y = 2;


// wasm is faster for small models
const sessionOption = { executionProviders: ['wasm', 'webgl'] };
let sess = undefined;
async function initOnnx() {
    sess = await ort.InferenceSession.create('./src/model.onnx', );//sessionOption);
}
initOnnx();

function canvasInit() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;
    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.strokeStyle = x;
    ctx.lineWidth = y;
    ctx.stroke();
    ctx.closePath();
}

function erase() {
    // var m = confirm("Want to clear");
    // if (m) {
    ctx.clearRect(0, 0, w, h);
    document.getElementById("canvasimg").style.display = "none";
    //}
}

function save() {
    document.getElementById("canvasimg").style.border = "2px solid";
    var dataURL = canvas.toDataURL();
    document.getElementById("canvasimg").src = dataURL;
    document.getElementById("canvasimg").style.display = "inline";
}

function findxy(res, e) {
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - canvas.offsetLeft;
        currY = e.clientY - canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
            ctx.beginPath();
            ctx.fillStyle = x;
            ctx.fillRect(currX, currY, 2, 2);
            ctx.closePath();
            dot_flag = false;
        }
    }
    if (res == 'up') {
        flag = false;
        makePrediciton(ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data);
    }
    if(res == "out") {
        flag = false;
    }
    if (res == 'move') {
        if (flag) {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;
            draw();
        }
    }
}

// ONNX CODE
async function makePrediciton(drawData) {
    //const sess = await ort.InferenceSession.create('./src/model.onnx');
    // const input = { input: new ort.Tensor('float32', inputData, [1, 3, 28, 28]) }
    
    const dims = [1, 4, 280, 280];
    // const size = dims[0] * dims[1] * dims[2] * dims[3];
    // const inputData = Float32Array.from({ length: size }, () => Math.random());

    // defining a dictionary
    const inputData = Float32Array.from(drawData);
    const feeds = { input: new ort.Tensor('float32', inputData, dims) };
    
    // console.log(feeds)

    const outputMap = await sess.run(feeds);
    const outputTensor = outputMap.output;
    
    plotData(outputTensor.data);
    console.log("prediction done");
    //console.log(`Output tensor: ${outputTensor.data}`);
}


// PLOTLY DATA
function plotData(probs) {
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    var trace = {
        x: labels,
        y: probs,
        type: "bar",
        hoverinfo: "none",
        opacity: 0.5,
        marker: {
            color: "red",
        },
    };

    var layout = {
        title: 'Digit Prediction',
        height: 400,
        width: 400,
        xaxis: {
            dtick: 1,
        },
        showlegend: false,
    };

    Plotly.newPlot('plotGraph', [trace], layout, {staticPlot: true});
}