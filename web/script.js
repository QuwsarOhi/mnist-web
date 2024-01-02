// wasm is faster for small models
const sessionOption = { executionProviders: ['wasm', 'webgl'] };
let sess = undefined;
async function initOnnx() {
    sess = await ort.InferenceSession.create('./src/model.onnx', );//sessionOption);
}
initOnnx();
plotData([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

const CANVAS_SIZE = 280;
var mousePressed = false;
var lastX, lastY;
var ctx;

function canvasInit() {
    ctx = document.getElementById('myCanvas').getContext("2d");

    $('#myCanvas').on('mousedown touchstart', function (e) {
        e.preventDefault();
        mousePressed = true;
        var touch = e.type === 'touchstart' ? e.originalEvent.touches[0] : e;
        Draw(touch.pageX - $(this).offset().left, touch.pageY - $(this).offset().top, false);
    });

    $('#myCanvas').on('mousemove touchmove', function (e) {
        e.preventDefault();
        if (mousePressed) {
            var touch = e.type === 'touchmove' ? e.originalEvent.touches[0] : e;
            Draw(touch.pageX - $(this).offset().left, touch.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas').on('mouseup touchend', function (e) {
        e.preventDefault();
        mousePressed = false;
        makePrediciton(ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data);
    });

    $('#myCanvas').on('mouseleave touchcancel', function (e) {
        e.preventDefault();
        mousePressed = false;
    });
}

function Draw(x, y, isDown) {
    if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = $('#selColor').val();
        ctx.lineWidth = 9; //$('#selWidth').val();
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x; lastY = y;
}

function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    plotData([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
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
            color: "cyan",
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