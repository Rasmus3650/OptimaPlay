var canvas, ctx, img, fps, interval, globalscale;
var player_chip_positions, player_data_positions, dice_positions, action_text_positions, bar_middle_pos, color_map, home_start_positions;
var player_data, curr_action_idx, roll_action_idx, counter, action_list, game_done, loaded_assets, current_chip_list, loaded_dice, curr_bar, curr_homes, end_counter;
var og_chip_width, new_chip_width, og_dice_width, new_dice_width, og_smallchip_width, new_smallchip_width, bigfont, mediumfont, smallfont;
window.onload = (event) => {
    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext("2d");
    og_chip_width = 33;
    og_dice_width = 38;
    og_smallchip_width = 11;
    end_counter = 3;
    curr_bar = [];
    curr_homes = [[], []];
    color_map = {0: "white", 1: "black"};

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
    curr_action_idx = 0
    roll_action_idx = -1
    counter = 0
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
    new_dice_width = Math.round(og_dice_width * scale);
    new_smallchip_width = Math.round(og_smallchip_width * scale)
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

    player_data_positions = {0: [Math.round(388 * scale), Math.round(16 * scale)],
                             1: [Math.round(388 * scale), Math.round(396 * scale)]}
    
    var dice_y = 200 - Math.round(og_dice_width / 2)
                            
    dice_positions =        {0: [Math.round(380 * scale), Math.round(dice_y * scale)],
                             1: [Math.round(440 * scale), Math.round(dice_y * scale)]}

    action_text_positions = {0: [Math.round(745 * scale), Math.round(90 * scale)],
                             1: [Math.round(1355 * scale), Math.round(145 * scale)]};
    
    home_start_positions = {0: [Math.round(564 * scale), Math.round(17 * scale)],
                            1: [Math.round(564 * scale), Math.round(372 * scale)]}

    bar_middle_pos = [Math.round(271 * scale), Math.round(181 * scale)]

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
    loaded_dice = [[new Image(), new Image(), new Image(), new Image(), new Image(), new Image()], [new Image(), new Image(), new Image(), new Image(), new Image(), new Image()]]
    loaded_assets[0].src = "/static/BackgammonAssets/White_chip.png"
    loaded_assets[1].src = "/static/BackgammonAssets/Black_chip.png"
    loaded_dice[0][0].src = "/static/Dice/white_1.png"
    loaded_dice[0][1].src = "/static/Dice/white_2.png"
    loaded_dice[0][2].src = "/static/Dice/white_3.png"
    loaded_dice[0][3].src = "/static/Dice/white_4.png"
    loaded_dice[0][4].src = "/static/Dice/white_5.png"
    loaded_dice[0][5].src = "/static/Dice/white_6.png"

    loaded_dice[1][0].src = "/static/Dice/black_1.png"
    loaded_dice[1][1].src = "/static/Dice/black_2.png"
    loaded_dice[1][2].src = "/static/Dice/black_3.png"
    loaded_dice[1][3].src = "/static/Dice/black_4.png"
    loaded_dice[1][4].src = "/static/Dice/black_5.png"
    loaded_dice[1][5].src = "/static/Dice/black_6.png"
    
    
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
            if (chip_n == 5) {
                break
            }
            if (field < 12) {
                ctx.drawImage(loaded_assets[current_chip_list[field][chip_n]], player_chip_positions[field][0], player_chip_positions[field][1] + (new_chip_width * chip_n), new_chip_width, new_chip_width);
            } else {
                ctx.drawImage(loaded_assets[current_chip_list[field][chip_n]], player_chip_positions[field][0], player_chip_positions[field][1] - (new_chip_width * chip_n), new_chip_width, new_chip_width);
            }
        }
    }
}

function draw_players_data() {
    ctx.font = mediumfont + "px Arial";
    ctx.fillStyle = "white";
    ctx.fillText("Player 0",player_data_positions[0][0],player_data_positions[0][1])
    ctx.fillStyle = "black";
    ctx.fillText("Player 1",player_data_positions[1][0],player_data_positions[1][1])
}


function create_action_list(ac_list) {
    action_list = Array(ac_list.length);
    ac_list.forEach(element => {
        action_list[element[0]] = [element[1], element[2], element[3], element[4], element[5], element[6]]
    });
    console.log(action_list)
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

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function draw_dice_roll() {
    roll_action_idx -= 1
    var player = action_list[curr_action_idx][0]
    var rng1 = Math.floor(Math.random() * 6)
    var rng2 = Math.floor(Math.random() * 6)
    ctx.drawImage(loaded_dice[player][rng1], dice_positions[0][0], dice_positions[0][1], new_dice_width, new_dice_width)
    ctx.drawImage(loaded_dice[player][rng2], dice_positions[1][0], dice_positions[1][1], new_dice_width, new_dice_width)
}

function draw_dice(player, rolls) {
    ctx.drawImage(loaded_dice[player][rolls[0] - 1], dice_positions[0][0], dice_positions[0][1], new_dice_width, new_dice_width)
    ctx.drawImage(loaded_dice[player][rolls[1] - 1], dice_positions[1][0], dice_positions[1][1], new_dice_width, new_dice_width)
}

function draw_bar() {
    var drawn_under = 0
    var drawn_above = 0
    for (var bar_idx = 0; bar_idx < curr_bar.length; bar_idx++) {
        if (bar_idx == 0) {
            ctx.drawImage(loaded_assets[curr_bar[bar_idx]], bar_middle_pos[0], bar_middle_pos[1], new_chip_width, new_chip_width);
        } else if (bar_idx % 2 == 0) {
            drawn_under += 1
            ctx.drawImage(loaded_assets[curr_bar[bar_idx]], bar_middle_pos[0], bar_middle_pos[1] + (drawn_under * new_chip_width), new_chip_width, new_chip_width);
        } else if (bar_idx % 2 == 1) {
            drawn_above += 1
            ctx.drawImage(loaded_assets[curr_bar[bar_idx]], bar_middle_pos[0], bar_middle_pos[1] - (drawn_above * new_chip_width), new_chip_width, new_chip_width);
        }
    }
}

function draw_chip_numbers() {
    ctx.font = smallfont + "px Arial";
    ctx.fillStyle = "red";
    var scale = globalscale / 100
    for (var i = 0; i < current_chip_list.length; i++) {
        if (current_chip_list[i].length > 0) {
            ctx.fillText(current_chip_list[i].length.toString(), player_chip_positions[i][0] + Math.round(13 * scale), player_chip_positions[i][1] + Math.round(20 * scale));
        }
    }
}

function draw_homes() {
    for (var p_id = 0; p_id < curr_homes.length; p_id++) {
        for (var chip_n = 0; chip_n < curr_homes[p_id].length; chip_n++) {
            if (p_id == 0) {
                ctx.drawImage(loaded_assets[p_id], home_start_positions[p_id][0], home_start_positions[p_id][1] + (chip_n * (new_smallchip_width + 1)), new_smallchip_width, new_smallchip_width)
            }
            if (p_id == 1) {
                ctx.drawImage(loaded_assets[p_id], home_start_positions[p_id][0], home_start_positions[p_id][1] - (chip_n * (new_smallchip_width + 1)), new_smallchip_width, new_smallchip_width)
            }
        }
    }
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    draw_player_chips();
    draw_chip_numbers();
    draw_players_data();
    draw_bar();
    draw_homes();
    if (curr_action_idx >= action_list.length) {
        if (end_counter == 0) {
            end_game()
        }
        end_counter -= 1
    } else {
        if (action_list[curr_action_idx][1] == "ROLL") {
            roll_action_idx = 3
            curr_action_idx += 1
        }
        if (roll_action_idx != -1) {
            draw_dice_roll()
            roll_action_idx -= 1
        } else {
            draw_dice(action_list[curr_action_idx][0], action_list[curr_action_idx][3])
            if (counter == 1) {
                current_chip_list = action_list[curr_action_idx][1]
                curr_bar = action_list[curr_action_idx][2]
                curr_homes = action_list[curr_action_idx][5]
            }
            if (counter == 2) {
                curr_action_idx += 1
                counter = 0
            }
            counter += 1
        }
    }
}

