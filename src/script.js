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


// function generate_image(image) {
//     // create an offscreen canvas
//     var canvas = document.createElement('canvas');
//     canvas.setAttribute("id", "canvas_id");
//     canvas.style.cssText = 'width: 280; height: 280; z-index: 1;';

//     document.getElementById('imageContainer').appendChild(canvas);
//     document.body.appendChild(canvas);

//     var ctx = canvas.getContext("2d");
    
//     if (image === undefined)
//       var images = generate_image_array_data();
    
//     // This is the pixel array of the image
//     var image = images[2];
    
//     var data = new Uint8ClampedArray( new Float64Array( image ).buffer );
//     var palette = new ImageData(data, 3, 4);
//     ctx.putImageData(palette, 0, 0);
    
//     ctx.imageSmootingEnabled = false;
//     ctx.globalCompositeOperation = "copy";
//     ctx.drawImage(canvas, 0, 0, 3, 4, 0, 0, canvas.width, canvas.height);
// }

// function generate_image_array_data() {
//     return [, , [-7.440151952041672e-45, -3.549412689007182e-44, -1.6725845544560632e-43, -7.785321621854041e-43, -3.579496378208359e-42, -1.6256392941914181e-41 ] ];
// }

//   window.onload = generate_image;