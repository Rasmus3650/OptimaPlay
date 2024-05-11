var canvas, ctx, globalscale;
window.onload = (event) => {
    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext("2d");
    
    //img = new Image();
    //img.src = "/static/.png";
    //ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    /*
    fps=2;
    var fpsSlider = document.getElementById("fpsSlider");
    var fpsValueDisplay = document.getElementById("fpsValue");
    let urlParams = new URLSearchParams(window.location.search);
    
    let fpsParam = urlParams.get('fps');
    if (fpsParam !== null) {
        fps = parseInt(fpsParam);
        fpsSlider.value = fps;
        fpsValueDisplay.textContent = fps;
    }

    fpsSlider.addEventListener('input', function(){
        fps = parseInt(this.value)
        fpsValueDisplay.textContent = this.value
        if (animation_is_running()) {
            clearInterval(interval);
            start_animation(fps);
        }
    })
    
    var canvasScaleSlider = document.getElementById("canvasScaleSlider");
    canvasScaleSlider.addEventListener("input", (event) => {
        change_scale(event.target.value);
      });

    let scaleParam = urlParams.get('scale');
    if (scaleParam == null){
        scaleParam = 50
    }
    globalscale = scaleParam
    */
    load_data();
    draw_map();
    //change_scale(parseInt(scaleParam), firstcall = true);
    /*
    let redirectParam = urlParams.get('redirect');

    if (redirectParam == "True") {

        let url = window.location.href;
        var u_score_idx = url.lastIndexOf("_");
        var q_mark_idx = url.lastIndexOf("?");
        var game_number = parseInt(url.slice(u_score_idx + 1, q_mark_idx));
        
        if (game_number != 0) {
            //pauseResume(force_start=true);
        }
        
    }
    */
};

function draw_map() {
    console.log("DRAWING MAP");
    var tileWidth = 10;
    var tileHeight = 10;
    var outlineWidth = 1; // Adjust this value for the thickness of the outline
    var outlineColor = "black"; // Adjust this value for the color of the outline

    for (var y = 0; y < json_data["map"]["map"].length; y++) {
        for (var x = 0; x < json_data["map"]["map"][y].length; x++) {
            // Draw filled rectangle for each tile

            tile = json_data["map"]["map"][y][x]['biome']
            if (tile === "Ocean") {
                ctx.fillStyle = "blue";
            } else if (tile === "Plains") {
                ctx.fillStyle = "green";
            } else if (tile === "Forest") {
                ctx.fillStyle = "darkgreen"; // Adjusted color for Forest
            } else if (tile === "Mountain") {
                ctx.fillStyle = "grey";
            } else if (tile === "Beach") {
                ctx.fillStyle = "yellow";
            } else {
                ctx.fillStyle = "white"; // Default color
            }
            ctx.fillRect(x * tileWidth, y * tileHeight, tileWidth, tileHeight);
            
            // Draw outline for each tile
            ctx.strokeStyle = outlineColor;
            ctx.lineWidth = outlineWidth;
            ctx.strokeRect(x * tileWidth, y * tileHeight, tileWidth, tileHeight);
        }
    }
}
    

function change_scale(scale, firstcall = false) {
    globalscale = scale;
    scale = scale / 100;
    bigfont = Math.round(32 * scale);
    mediumfont = Math.round(20 * scale);
    smallfont = Math.round(18 * scale);

    canvas.width = Math.round(1600 * scale);
    canvas.height = Math.round(900 * scale);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (firstcall == false) {
        /*
        draw_player_cards(loaded_player_card_dict);
        draw_dealer_cards();
        draw_players_data();
        draw_current_bets();
        draw_folded();
        draw_cards_on_table();
        */
    }
    
}

function print(stuff, title) {
    console.log("__________________________________");
    console.log(title + ": ");
    console.log(stuff);
    console.log("__________________________________\n");
}

function start_animation(fps) {
    interval = setInterval(()=>{//
        animate();
    },1000/fps);
}

function get_redirect() {
    let url_params = new URLSearchParams(window.location.search);
    let redirect = url_params.get('redirect');
    if (redirect === 'True') {
        return true;
    } else {
        return false;
    } 
}


function load_data() {
    let data = document.getElementById("game_data").innerHTML.toString()
    json_data = JSON.parse(data)
    var keys = Object.keys(json_data)
    keys.forEach(element => {
        json_data[element] = JSON.parse(json_data[element])
    });

    console.log(json_data)
}

function animation_is_running() {
    var buttonInput = document.getElementById("toggle_game");
    return buttonInput.textContent === "PAUSE GAME";
}

function pauseResume(force_start = false) {
    var buttonInput = document.getElementById("toggle_game");
    let button_state;
    if (force_start == false) {
        button_state = (buttonInput.textContent === "PAUSE GAME") ? "START GAME" : "PAUSE GAME";
    } else {
        button_state = "PAUSE GAME";
    }
    
    buttonInput.textContent = button_state;
    if (button_state == "PAUSE GAME") {
        start_animation(fps);
    } else {
        clearInterval(interval);
    }
}


function redirect_to_next_game(game){
    let url = window.location.href;
    let last_idx = url.lastIndexOf("/");
    if (last_idx !== -1) {
        let baseUrl = url.substring(0, last_idx + 1); 
        let result = baseUrl + "Game_" + game + "?redirect=True&fps=" + fps + "&scale=" + globalscale;
        window.location.href = result;
    }
}

function end_game() {
    //clearInterval(interval);
    var redirect = get_redirect();
    var url = window.location.href;
    if (redirect) {
        var u_score_idx = url.lastIndexOf("_");
        var q_mark_idx = url.lastIndexOf("?");
        var newgame = parseInt(url.slice(u_score_idx + 1, q_mark_idx)) + 1;
        redirect_to_next_game(newgame);
    }
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    //draw_current_bets();
    //draw_folded();
    //draw_cards_on_table();

    
}

