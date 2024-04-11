var canvas, ctx, x, y, dx, radius, img, fps, interval, actions, cards, initbals, postgamebals, winners, metadata, log;
window.onload = (event) => {
    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext("2d");

    // Define the initial position and velocity of the circle
    x = 50;
    y = canvas.height / 2;
    dx = 2; // Change in x (horizontal speed)
    radius = 20;
    img = new Image();
    img.src = "/static/Poker_table.png";
    ctx.drawImage(img, 10, 10);
    fps=1;
    var fpsSlider = document.getElementById("fpsSlider");
    var fpsValueDisplay = document.getElementById("fpsValue");
    fpsSlider.addEventListener('input', function(){
        fps = parseInt(this.value)
        fpsValueDisplay.textContent = this.value
        clearInterval(interval);
        interval = setInterval(()=>{//
            animate();
        },1000/fps);
    })
    load_data();
};

function load_data() {
    let data = document.getElementById("game_data").innerText;
    [actions, cards, initbals, log, metadata, postgamebals, winners] = data.split("', '");
    console.log(actions);
}

function pauseResume() {
    var buttonInput = document.getElementById("toggle_game");
    let button_state = (buttonInput.textContent === "PAUSE GAME") ? "START GAME" : "PAUSE GAME";
    buttonInput.textContent = button_state;
    if (button_state == "PAUSE GAME") {
        interval = setInterval(()=>{//
            animate();
        },1000/fps);
    } else {
        clearInterval(interval);
    }
    console.log(actions);
}

// Function to draw the circle
function drawCircle() {
    
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas

    ctx.drawImage(img, 0, 0);
    
    
    
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = "blue";
    ctx.fill();
    ctx.closePath();
}

// Function to update the position of the circle
function updatePosition() {
    x += dx; // Move the circle horizontally
    if (x + radius > canvas.width || x - radius < 0) {
        // Reverse the direction if the circle reaches the canvas boundary
        dx = -dx;
    }
}

function animate() {
    drawCircle(); // Draw the circle
    updatePosition(); // Update the position
    //requestAnimationFrame(animate); // Call animate() recursively to create animation
}

