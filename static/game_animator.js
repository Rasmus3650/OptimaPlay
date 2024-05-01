var canvas, ctx, img, fps, interval, actions, cards, initbals, postgamebals, winners, metadata, log, globalscale;
var player_card_positions, card_offset, dealer_chip_positions, player_data_positions, player_bet_positions, table_card_positions, pot_pos, action_text_positions;
var dealer_chip_img, pot, player_data, cross_img, curr_action_idx, curr_action_subcounter, action_list, loaded_player_card_dict, loaded_table_cards_list, cards_on_table, dealer, game_done;
var xscale, yscale, og_card_width, og_card_height, new_card_width, new_card_height, bigfont, mediumfont, smallfont, player_data_offset;
window.onload = (event) => {
    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext("2d");
    og_card_width = 75;
    og_card_height = 109;
    /*
    var c_width = canvas.width;
    var c_height = canvas.height;
    

    xscale = c_width / 1600;
    yscale = c_height / 900;
    bigfont = 32 * Math.min(xscale, yscale);
    mediumfont = 20 * Math.min(xscale, yscale);
    smallfont = 18 * Math.min(xscale, yscale);
    player_data_offset = 25 * yscale;
    console.log("xscale: " + xscale);
    console.log("yscale: " + yscale);
    
    new_card_width = og_card_width * xscale;
    new_card_height = og_card_height * yscale;
    */

    dealer_chip_img = new Image();
    dealer_chip_img.src = "/static/Dealer_brick.png";

    img = new Image();
    img.src = "/static/Poker_table.png";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    cross_img = new Image();
    cross_img.src = "/static/cross.png";

    cards_on_table = 0;

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

    loaded_player_card_dict = {};
    loaded_table_cards_list = [];

    curr_action_idx = 0;
    curr_action_subcounter = 0;

    /*
    player_card_positions = {0: [720, 140], 1: [1180, 190], 2: [1180, 510], 3: [720, 560], 4: [260, 510], 5: [260, 190]}
    dealer_chip_positions = {0: [670, 140], 1: [1200, 150], 2: [1350, 510], 3: [885, 565], 4: [215, 520], 5: [215, 220]}
    player_data_positions = {0: [770, 30], 1: [1380, 85], 2: [1460, 660], 3: [770, 735], 4: [90, 660], 5: [160, 85]}
    player_bet_positions = {0: [790, 280], 1: [1165, 320], 2: [1165, 505], 3: [790, 555], 4: [410, 505], 5: [410, 320]}
    
    

    player_card_positions = {0: [720 * xscale, 140 * yscale], 1: [1180 * xscale, 190 * yscale], 2: [1180 * xscale, 510 * yscale], 3: [720 * xscale, 560 * yscale], 4: [260 * xscale, 510 * yscale], 5: [260 * xscale, 190 * yscale]}
    dealer_chip_positions = {0: [670 * xscale, 140 * yscale], 1: [1200 * xscale, 150 * yscale], 2: [1350 * xscale, 510 * yscale], 3: [885 * xscale, 565 * yscale], 4: [215 * xscale, 520 * yscale], 5: [215 * xscale, 220 * yscale]}
    player_data_positions = {0: [770 * xscale, 30 * yscale], 1: [1380 * xscale, 85 * yscale], 2: [1460 * xscale, 660 * yscale], 3: [770 * xscale, 735 * yscale], 4: [90 * xscale, 660 * yscale], 5: [160 * xscale, 85 * yscale]}
    player_bet_positions = {0: [790 * xscale, 280 * yscale], 1: [1165 * xscale, 320 * yscale], 2: [1165 * xscale, 505 * yscale], 3: [790 * xscale, 555 * yscale], 4: [410 * xscale, 505 * yscale], 5: [410 * xscale, 320 * yscale]}

    var table_cards_y = 360 * yscale;
    table_card_positions = {0: [602 * xscale, table_cards_y], 1: [682 * xscale, table_cards_y], 2: [762 * xscale, table_cards_y], 3: [842 * xscale, table_cards_y], 4: [922 * xscale, table_cards_y]}
    card_offset = new_card_width + 5;

    action_text_positions = {0: [745 * xscale, 90 * yscale], 1: [1355 * xscale, 145 * yscale], 2: [1440 * xscale, 720 * yscale], 3: [740 * xscale, 795 * yscale], 4: [60 * xscale, 720 * yscale], 5: [125 * xscale, 145 * yscale]};

    pot_pos = [740 * xscale, 350 * yscale];
    */
    pot = 0;

    player_data = {0: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   1: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   2: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   3: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   4: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   5: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0}};

    let scaleParam = urlParams.get('scale');
    if (scaleParam == null){
        scaleParam = 50
    }
    globalscale = scaleParam
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
    player_data_offset = Math.round(25 * scale);
    new_card_width = Math.round(og_card_width * scale);
    new_card_height = Math.round(og_card_height * scale);
    player_card_positions = {0: [Math.round(720 * scale), Math.round(140 * scale)],
                             1: [Math.round(1180 * scale), Math.round(190 * scale)],
                             2: [Math.round(1180 * scale), Math.round(510 * scale)],
                             3: [Math.round(720 * scale), Math.round(560 * scale)],
                             4: [Math.round(260 * scale), Math.round(510 * scale)],
                             5: [Math.round(260 * scale), Math.round(190 * scale)]}

    dealer_chip_positions = {0: [Math.round(670 * scale), Math.round(140 * scale)],
                             1: [Math.round(1200 * scale), Math.round(150 * scale)],
                             2: [Math.round(1350 * scale), Math.round(510 * scale)],
                             3: [Math.round(885 * scale), Math.round(565 * scale)],
                             4: [Math.round(215 * scale), Math.round(520 * scale)],
                             5: [Math.round(215 * scale), Math.round(220 * scale)]}

    player_data_positions = {0: [Math.round(770 * scale), Math.round(30 * scale)],
                             1: [Math.round(1380 * scale), Math.round(85 * scale)],
                             2: [Math.round(1460 * scale), Math.round(660 * scale)],
                             3: [Math.round(770 * scale), Math.round(735 * scale)],
                             4: [Math.round(90 * scale), Math.round(660 * scale)],
                             5: [Math.round(160 * scale), Math.round(85 * scale)]}

    player_bet_positions = {0: [Math.round(790 * scale), Math.round(280 * scale)],
                            1: [Math.round(1165 * scale), Math.round(320 * scale)],
                            2: [Math.round(1165 * scale), Math.round(505 * scale)],
                            3: [Math.round(790 * scale), Math.round(555 * scale)],
                            4: [Math.round(410 * scale), Math.round(505 * scale)],
                            5: [Math.round(410 * scale), Math.round(320 * scale)]}

    var table_cards_y = Math.round(360 * scale);
    table_card_positions = {0: [Math.round(602 * scale), table_cards_y],
                            1: [Math.round(682 * scale), table_cards_y],
                            2: [Math.round(762 * scale), table_cards_y],
                            3: [Math.round(842 * scale), table_cards_y],
                            4: [Math.round(922 * scale), table_cards_y]}

    card_offset = new_card_width + 5;

    action_text_positions = {0: [Math.round(745 * scale), Math.round(90 * scale)],
                             1: [Math.round(1355 * scale), Math.round(145 * scale)],
                             2: [Math.round(1440 * scale), Math.round(720 * scale)],
                             3: [Math.round(740 * scale), Math.round(795 * scale)],
                             4: [Math.round(60 * scale), Math.round(720 * scale)],
                             5: [Math.round(125 * scale), Math.round(145 * scale)]};

    pot_pos = [Math.round(740 * scale), Math.round(350 * scale)];
    canvas.width = Math.round(1600 * scale);
    canvas.height = Math.round(900 * scale);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (firstcall == false) {
        draw_player_cards(loaded_player_card_dict);
        draw_players_data();
        draw_current_bets();
        draw_pot();
        draw_folded();
        draw_cards_on_table();
        draw_dealer_chip();
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

function load_cards(card_dict, table_cards) {
    loaded_player_card_dict = {};
    loaded_table_cards_list = [];
    var p_keys = Object.keys(card_dict);
    for (var i = 0; i < p_keys.length; i++) {
        loaded_player_card_dict[p_keys[i]] = [new Image(), new Image()]
        loaded_player_card_dict[p_keys[i]][0].src = "/static/" + get_card_filename(card_dict[p_keys[i]][0]);
        loaded_player_card_dict[p_keys[i]][1].src = "/static/" + get_card_filename(card_dict[p_keys[i]][1]);
    }
    for (var i = 0; i < table_cards.length; i++) {
        loaded_table_cards_list.push(new Image());
        loaded_table_cards_list[loaded_table_cards_list.length - 1].src = "/static/" + get_card_filename(table_cards[i]);
    }

}

function set_bals(bals_dict) {
    var p_keys = Object.keys(bals_dict);
    for (var i = 0; i < p_keys.length; i++) {
        player_data[p_keys[i]]["bal"] = bals_dict[p_keys[i]];
    }
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
    console.log(json_data)
    create_action_list(json_data['actions']);
    set_bals(json_data['init_bals'])
    load_cards(json_data['cards']['player_hands'],json_data['cards']['cards_on_table'])
    dealer = json_data['metadata']['dealer']
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

function get_card_filename(card) {
    var rank_str = card[0].toString();
    if (card[0] > 10) {
        if (card[0] == 11) { rank_str = "jack"; }
        if (card[0] == 12) { rank_str = "queen"; }
        if (card[0] == 13) { rank_str = "king"; }
        if (card[0] == 14) { rank_str = "ace"; }
    }
    var filename = rank_str + "_of_" + card[1].toString().toLowerCase() + ".png";
    return filename;
}


function draw_player_cards(p_card_dict) {
    var keys = Object.keys(p_card_dict);
    for (var i = 0; i < keys.length; i++) {
        for (var j = 0; j < p_card_dict[keys[i]].length; j++) {
            var x = parseInt(player_card_positions[keys[i]][0]) + parseInt(j) * card_offset;
            var y = parseInt(player_card_positions[keys[i]][1]);
            ctx.drawImage(p_card_dict[keys[i]][j], x, y, new_card_width, new_card_height);
        }
    }
}

function draw_players_data() {
    var keys = Object.keys(player_data);
    for (var i = 0; i < keys.length; i++){
        var x = parseInt(player_data_positions[keys[i]][0])
        var y = parseInt(player_data_positions[keys[i]][1]+5)
        ctx.font = bigfont + "px Arial";
        ctx.fillStyle = "white"; 
        ctx.fillText("P "+keys[i],x,y)
        ctx.font = mediumfont + "px Arial";
        var x_bal = parseInt(player_data_positions[keys[i]][0])
        var y_bal = parseInt(player_data_positions[keys[i]][1]+player_data_offset)
        ctx.fillText(player_data[keys[i]]["bal"] + " $", x_bal, y_bal)
    }
}


function draw_dealer_chip() {
    ctx.drawImage(dealer_chip_img, dealer_chip_positions[dealer][0], dealer_chip_positions[dealer][1], dealer_chip_img.width * (globalscale / 100), dealer_chip_img.height * (globalscale / 100));
}

function draw_cards_on_table() {
    for (var i = 0; i < cards_on_table; i++) {
        ctx.drawImage(loaded_table_cards_list[i], table_card_positions[i][0], table_card_positions[i][1], new_card_width, new_card_height);
    }
}

function draw_pot() {
    ctx.font = bigfont + "px Arial";
    ctx.fillStyle = "white"; 
    ctx.fillText("Pot: " + pot + " $", pot_pos[0], pot_pos[1]);
}

function draw_current_bets() {
    var p_ids = Object.keys(player_data);
    ctx.font = smallfont + "px Arial";
    ctx.fillStyle = "white"; 
    for (var i = 0; i < p_ids.length; i++) {
        ctx.fillText(player_data[p_ids[i]]["curr_bet"] + " $", player_bet_positions[p_ids[i]][0], player_bet_positions[p_ids[i]][1]);
    }
}

function display_action(action) {
    ac_str = action[1];
    if (action[2] > 0.0) {
        ac_str += " " + action[2] + " $";
    }
    ctx.font = "bold " + smallfont + "px Arial";
    ctx.fillStyle = "green"; 
    ctx.fillText(ac_str, action_text_positions[action[0]][0], action_text_positions[action[0]][1]);
}

function draw_cross(player_id){
    ctx.drawImage(cross_img, player_card_positions[player_id][0], player_card_positions[player_id][1], cross_img.width * (globalscale / 100), cross_img.height * (globalscale / 100));
}

function draw_folded() {
    var p_ids = Object.keys(player_data);
    for (var i = 0; i < p_ids.length; i++) {
        if (player_data[p_ids[i]]["folded"]) {
            draw_cross(p_ids[i]);
        }
    }
}


function do_action(action) {
    var p_id = action[0]
    if (action[1] == "Fold") {
        player_data[p_id]['folded'] = true;
    } else if (action[1] == "Raise" || action[1] == "Call") {
        player_data[p_id]["bal"] = Math.round(((player_data[p_id]["bal"] - action[2]) + Number.EPSILON) * 100) / 100;
        player_data[p_id]["curr_bet"] = Math.round(((player_data[p_id]["curr_bet"] + action[2]) + Number.EPSILON) * 100) / 100;

    }
}

function create_action_list(ac_dict) {
    action_list = [["TRANS", "Pre-flop"]];
    var keys = Object.keys(ac_dict);
    for (var i = 0; i < keys.length; i++) {
        for (var j = 0; j < ac_dict[keys[i]].length; j++) {
            action_list.push(ac_dict[keys[i]][j]);
        }
        if (keys[i] != keys[keys.length - 1]) {
            if (ac_dict[keys[i + 1]].length > 0) {
                action_list.push(["TRANS", keys[i + 1]])
                var toDeal = 0;
                if (keys[i + 1] == "Flop") { toDeal = 3;}
                if (keys[i + 1] == "Turn" || keys[i + 1] == "River") { toDeal = 1;}
                for (var x = 0; x < toDeal; x++) {
                    action_list.push(["DEAL", "TABLE"]);
                }
            }
        }
    }
}

function collect_pot() {
    var p_ids = Object.keys(player_data);
    for (var i = 0; i < p_ids.length; i++) {
        let bet = player_data[p_ids[i]]['curr_bet']
        player_data[p_ids[i]]['curr_bet'] = 0;
        player_data[p_ids[i]]['total_bet'] = Math.round(((player_data[p_ids[i]]['total_bet'] +  bet) + Number.EPSILON) * 100) / 100;
        pot = Math.round(((pot + bet) + Number.EPSILON) * 100) / 100;
    }
}

function deal_card_to_table() {
    cards_on_table += 1;
}

function get_sidepots(log){
    let lines = log.split('\n');
    tot_pot_idx = lines.indexOf('Total pot:');
    let remaining_lines = lines.slice(tot_pot_idx+1);
    let player_amounts = {};
    remaining_lines.forEach(line => {
        if (line.startsWith("Adding")) {
            const parts = line.split("to player ");
            const playerId = parseInt(parts[1].split(' ')[0]);
            const amount = parseFloat(parts[0].split(' ')[1]);
            player_amounts[playerId] = amount;
        }
    });
    return player_amounts
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
    draw_player_cards(loaded_player_card_dict);
    draw_players_data();
    draw_current_bets();
    draw_pot();
    draw_folded();
    draw_cards_on_table();
    draw_dealer_chip();

    if (action_list.length <= curr_action_idx) {
        done = true;
        end_game();
    } else {
        var action = action_list[curr_action_idx];
        if (action[0] == "TRANS") {
            document.getElementById("gamestate").innerText = "Game state: " + action[1];
            curr_action_subcounter = 2;
            collect_pot();
        } else if (action[0] == "DEAL") {
            deal_card_to_table();
            curr_action_subcounter = 2;
        }else {
            display_action(action);
            curr_action_subcounter = (curr_action_subcounter + 1) % 3;
        }
        
        if (curr_action_subcounter == 0){ 
            curr_action_subcounter += 1;
        } else if (curr_action_subcounter == 1){
            do_action(action);
            curr_action_subcounter += 1;
        } else if (curr_action_subcounter == 2){
            curr_action_idx += 1;
            curr_action_subcounter = 0;
        }
    }
}

