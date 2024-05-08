var canvas, ctx, img, fps, interval, globalscale;
var player_chip_positions, player_data_positions, action_text_positions;
var player_data, curr_action_idx, action_list, game_done, loaded_assets, current_chip_list;
var og_chip_width, new_chip_width, bigfont, mediumfont, smallfont;
window.onload = (event) => {
    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext("2d");
    og_chip_width = 33;

    current_chip_list = [[1, 1], [], [], [], [], [0, 0, 0, 0, 0], [], [0, 0, 0], [], [], [], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [], [], [], [1, 1, 1], [], [1, 1, 1, 1, 1], [], [], [], [], [0, 0]];
    img = new Image();
    img.src = "/static/BackgammonAssets/Backgammon_table.png";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    game_done = false;

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
    player_data = {}
    load_data();
    change_scale(parseInt(scaleParam), firstcall = true);

    let redirectParam = urlParams.get('redirect');

    if (redirectParam == "True") {

        let url = window.location.href;
        var u_score_idx = url.lastIndexOf("_");
        var q_mark_idx = url.lastIndexOf("?");
        var game_number = parseInt(url.slice(u_score_idx + 1, q_mark_idx));
        
        if (game_number != 0) {
            pauseResume(force_start=true);
        }
        
    }
};

function change_scale(scale, firstcall = false) {
    globalscale = scale;
    scale = scale / 100;
    bigfont = Math.round(32 * scale);
    mediumfont = Math.round(20 * scale);
    smallfont = Math.round(18 * scale);
    new_chip_width = Math.round(og_chip_width * scale);
    var top_y = 20;
    var bot_y = 400 - 36 - 16 - 1;
    
    player_chip_positions = {0: [Math.round(522 * scale), Math.round(top_y * scale)],
                             1: [Math.round(478 * scale), Math.round(top_y * scale)],
                             2: [Math.round(435 * scale), Math.round(top_y * scale)],
                             3: [Math.round(391 * scale), Math.round(top_y * scale)],
                             4: [Math.round(348 * scale), Math.round(top_y * scale)],
                             5: [Math.round(304 * scale), Math.round(top_y * scale)],

                             6: [Math.round(244 * scale), Math.round(top_y * scale)],
                             7: [Math.round(200 * scale), Math.round(top_y * scale)],
                             8: [Math.round(157 * scale), Math.round(top_y * scale)],
                             9: [Math.round(113 * scale), Math.round(top_y * scale)],
                             10: [Math.round(70 * scale), Math.round(top_y * scale)],
                             11: [Math.round(26 * scale), Math.round(top_y * scale)],

                             12: [Math.round(26 * scale), Math.round(bot_y * scale)],
                             13: [Math.round(70 * scale), Math.round(bot_y * scale)],
                             14: [Math.round(113 * scale), Math.round(bot_y * scale)],
                             15: [Math.round(157 * scale), Math.round(bot_y * scale)],
                             16: [Math.round(200 * scale), Math.round(bot_y * scale)],
                             17: [Math.round(244 * scale), Math.round(bot_y * scale)],

                             18: [Math.round(304 * scale), Math.round(bot_y * scale)],
                             19: [Math.round(348 * scale), Math.round(bot_y * scale)],
                             20: [Math.round(391 * scale), Math.round(bot_y * scale)],
                             21: [Math.round(435 * scale), Math.round(bot_y * scale)],
                             22: [Math.round(478 * scale), Math.round(bot_y * scale)],
                             23: [Math.round(522 * scale), Math.round(bot_y * scale)]}

    player_data_positions = {0: [Math.round(1340 * scale), Math.round(240 * scale)],
                             1: [Math.round(1140 * scale), Math.round(390 * scale)]}


    action_text_positions = {0: [Math.round(745 * scale), Math.round(90 * scale)],
                             1: [Math.round(1355 * scale), Math.round(145 * scale)]};

    canvas.width = Math.round(580 * scale);
    canvas.height = Math.round(400 * scale);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (firstcall == false) {
        draw_player_chips();
        draw_players_data();
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


function load_images() {
    loaded_assets = [new Image(), new Image()]
    loaded_assets[0].src = "/static/BackgammonAssets/White_chip.png"
    loaded_assets[1].src = "/static/BackgammonAssets/Black_chip.png"
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
    let data = document.getElementById("game_data").innerHTML.toString();
    json_data = JSON.parse(data);
    console.log(json_data);
    create_action_list(json_data['actions']);
    load_images();
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


function draw_player_chips() {
    for (var field = 0; field < current_chip_list.length; field++) {
        for (var chip_n = 0; chip_n < current_chip_list[field].length; chip_n++) {
            if (field < 12) {
                ctx.drawImage(loaded_assets[current_chip_list[field][chip_n]], player_chip_positions[field][0], player_chip_positions[field][1] + (new_chip_width * chip_n), new_chip_width, new_chip_width);
            } else {
                ctx.drawImage(loaded_assets[current_chip_list[field][chip_n]], player_chip_positions[field][0], player_chip_positions[field][1] - (new_chip_width * chip_n), new_chip_width, new_chip_width);
            }
            //ctx.drawImage(loaded_assets[current_chip_list[field][chip_n]], player_chip_positions[field][0], player_chip_positions[field][1], new_chip_width, new_chip_width);
        }
    }
}

function draw_players_data() {
    ctx.font = bigfont + "px Arial";
    ctx.fillStyle = "white";
    ctx.fillText("Player 0",player_data_positions[0][0],player_data_positions[0][1])
    ctx.fillText("Player 1",player_data_positions[1][0],player_data_positions[1][1])
}


function create_action_list(ac_list) {
    action_list = Array(ac_list.length);
    ac_list.forEach(element => {
        action_list[element[0]] = [element[1], element[2], element[3]]
    });
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
    clearInterval(interval);
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
    draw_player_chips();
    draw_players_data();
    draw_dealer_cards();
    
    //draw_current_bets();
    //draw_folded();
    //draw_cards_on_table();

    
}

