
// Canvas and its context
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");

// Button for clearing the canvas
const clearButton = document.getElementById("clear");
// Button for sending bitmap data to lambda to guess number
const guessButton = document.getElementById("guess")

const guessText = document.getElementById("guess_text")

// Canvas Dimensions
canvas.width = 420;
canvas.height = 420;

// Scale down factor to reach target input size of image
const SCALE_FACTOR = 15;

// Flag for if currently drawing
let is_drawing = false;

// Brush parameters
let draw_width = "12";
let draw_color = "black";

// Event listeners for drawing
canvas.addEventListener("mousedown", start);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stop);

// Clear button click
clearButton.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    guessText.innerHTML = "Guess: ";
});

// Guess button click
guessButton.addEventListener("click", processImage)

// Start drawing when mouse button is clicked
function start(event){
    is_drawing = true;

    // Brush parameters
    ctx.strokeStyle = draw_color;
    ctx.lineWidth = draw_width;
    ctx.lineCap = "round";
    
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    event.preventDefault();
}


// Stop drawing when mouse button is lifted
function stop(){
    is_drawing = false;
}

// Draw while mmouse button is held and dragged
function draw(event){
    if(!is_drawing){return;}
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
}


// Scales image into proper input for model:
// 28x28 grey scale image
async function processImage(){
    imageD = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = [];
    let j = 0;
    
    // Retrieve grey scale information from ImageData
    for(let i = 3; i < imageD.data.length; i+=4){
        pixels[j] = imageD.data[i];
        j++;
    }

    // Format pixels into 2D array
    const pixels_2d = []

    for(let i = 0; i < Math.floor(pixels.length/canvas.width); i++){
        pixels_2d.push(pixels.slice(canvas.width*i, canvas.width*(i+1)));
    }

    // Scale down 420x420 image to 28x28 by 
    // taking average of 15x15 pixels

    const scale_pixels = [];
    const new_dim = Math.floor(pixels_2d.length/SCALE_FACTOR);
    let new_dim_x = 0;
    let new_dim_y = 0;

    for (let i = 0; i < new_dim; i++){
        new_dim_y = SCALE_FACTOR*i;
        const scale_temp = []
        for (let j = 0; j < new_dim; j++){
            new_dim_x = SCALE_FACTOR*j;
            scale_temp[j] = avg([new_dim_x, new_dim_y], pixels_2d);
        }

        scale_pixels.push(scale_temp);
    }

    sendImageToModel(scale_pixels);
}

async function sendImageToModel(pixels_2d) {
    let data = JSON.stringify(pixels_2d);
    data = "[[" + data + "]]"
    let payload = {
        "data" : data
    }
    let response = await fetch("https://sbr23ay7m6.execute-api.us-east-1.amazonaws.com/default/numberGuesserFunction", {
        method: 'POST',
        mode: 'cors',
        body: JSON.stringify(payload),
        dataType: 'json',
        headers: {
          'Content-Type': 'application/json'
        }
    });
    let jsonResponse = await response.json();
    let guess = jsonResponse.body;
    guessText.innerHTML = "Guess: " + guess;
    console.log(jsonResponse);
}

// Takes average of 15x15 pixels given a starting index
function avg(startIn, arr){
    let averg = 0;
    for(let i = startIn[1]; i < startIn[1] + SCALE_FACTOR; i++){
        for(let j = startIn[0]; j < startIn[0] + SCALE_FACTOR; j++){
            averg += arr[i][j];
        }
    }
    return Math.floor(averg/(SCALE_FACTOR*SCALE_FACTOR));
}