var canvas, ctx, img, fps, interval, globalscale;
var player_card_positions, card_offset, player_data_positions, player_bet_positions, dealer_card_positions, action_text_positions, hand_offset,card_width_offset;
var player_data, cross_img, curr_action_idx, curr_action_subcounter, action_list, loaded_player_card_dict, loaded_dealer_cards_list, dealer_cards, game_done;
var xscale, yscale, og_card_width, og_card_height, new_card_width, new_card_height, bigfont, mediumfont, smallfont, player_data_offset;
window.onload = (event) => {
    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext("2d");
    og_card_width = 75;
    og_card_height = 109;


    img = new Image();
    img.src = "/static/BlackjackAssets/Blackjack_table.png";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    cross_img = new Image();
    cross_img.src = "/static/misc/cross.png";

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
    loaded_dealer_cards_list = [];

    curr_action_idx = 0;
    curr_action_subcounter = 0;

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
    player_data_offset = Math.round(25 * scale);
    new_card_width = Math.round(og_card_width * scale);
    new_card_height = Math.round(og_card_height * scale);
    
    player_card_positions = {0: [Math.round(1330 * scale), Math.round(380 * scale)],
                             1: [Math.round(1130 * scale), Math.round(520 * scale)],
                             2: [Math.round(875 * scale), Math.round(600 * scale)],
                             3: [Math.round(640 * scale), Math.round(600 * scale)],
                             4: [Math.round(380 * scale), Math.round(520 * scale)],
                             5: [Math.round(180 * scale), Math.round(380 * scale)]}

    player_data_positions = {0: [Math.round(1340 * scale), Math.round(240 * scale)],
                             1: [Math.round(1140 * scale), Math.round(390 * scale)],
                             2: [Math.round(890 * scale), Math.round(470 * scale)],
                             3: [Math.round(660 * scale), Math.round(470 * scale)],
                             4: [Math.round(410 * scale), Math.round(390 * scale)],
                             5: [Math.round(200 * scale), Math.round(240 * scale)]}

    player_bet_positions = {0: [Math.round(790 * scale), Math.round(280 * scale)], //Not correct
                            1: [Math.round(1165 * scale), Math.round(320 * scale)],
                            2: [Math.round(1165 * scale), Math.round(505 * scale)],
                            3: [Math.round(790 * scale), Math.round(555 * scale)],
                            4: [Math.round(410 * scale), Math.round(505 * scale)],
                            5: [Math.round(410 * scale), Math.round(320 * scale)]}

    var dealer_cards_y = Math.round(100 * scale);
    dealer_card_positions = [Math.round(700 * scale), dealer_cards_y]

    card_offset = Math.round(new_card_height / 3);
    card_width_offset = new_card_width+5
    hand_offset = new_card_width + 5;

    action_text_positions = {0: [Math.round(745 * scale), Math.round(90 * scale)], //Not correct
                             1: [Math.round(1355 * scale), Math.round(145 * scale)],
                             2: [Math.round(1440 * scale), Math.round(720 * scale)],
                             3: [Math.round(740 * scale), Math.round(795 * scale)],
                             4: [Math.round(60 * scale), Math.round(720 * scale)],
                             5: [Math.round(125 * scale), Math.round(145 * scale)]};

    canvas.width = Math.round(1600 * scale);
    canvas.height = Math.round(900 * scale);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (firstcall == false) {
        draw_player_cards(loaded_player_card_dict);
        draw_dealer_cards();
        draw_players_data();
        draw_current_bets();
        draw_folded();
        draw_cards_on_table();
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

function load_cards(player_cards, dealer_cards) {
    loaded_player_card_dict = {};
    loaded_dealer_cards_list = [];
    var p_keys = Object.keys(player_cards);
    for (var i = 0; i < p_keys.length; i++) {
        var p_id = p_keys[i]
        loaded_player_card_dict[p_id] = {}
        var hands_dict = player_cards[p_id]
        var hand_amount = Object.keys(hands_dict).length
        for (var hand_id = 0; hand_id < hand_amount; hand_id++) {
            var temp_hand_lst = []
            var cards_on_hand = hands_dict[hand_id].length
            for (var card_id = 0; card_id < cards_on_hand; card_id++) {
                temp_hand_lst.push(new Image())
                temp_hand_lst[temp_hand_lst.length - 1].src = "/static/Cards/" + get_card_filename([player_cards[p_id][hand_id][card_id]["current_rank"], player_cards[p_id][hand_id][card_id]["current_suit"]])
            }
            loaded_player_card_dict[p_id][hand_id] = temp_hand_lst
        }
    }

    for (var i = 0; i < dealer_cards.length; i++) {
        loaded_dealer_cards_list.push(new Image());
        loaded_dealer_cards_list[loaded_dealer_cards_list.length - 1].src = "/static/Cards/" + get_card_filename([dealer_cards[i]["current_rank"], dealer_cards[i]["current_suit"]]);
    }
}

function set_bals(bals_dict) {
    var start_bals = bals_dict["start_bal"]
    var p_keys = Object.keys(start_bals);
    for (var i = 0; i < p_keys.length; i++) {
        player_data[p_keys[i]] = {"bal": start_bals[p_keys[i]]};
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
    set_bals(json_data['bals']);
    create_action_list(json_data['actions']);
    load_cards(json_data['cards']['player_cards'],json_data['cards']['dealer_cards'])
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
    console.log("DRAWING PLAYER CARDS (" + keys + ")")
    for (var i = 0; i < keys.length; i++) {
        var hand_ids = Object.keys(p_card_dict[keys[i]])
        console.log("PLAYER: " + keys[i])
        console.log(hand_ids)
        for (var hand_id = 0; hand_id < hand_ids.length; hand_id++) {
            console.log("    HAND ID: " + hand_id)
            console.log("    Hand: " + p_card_dict[keys[i]][hand_id])
            for (var j = 0; j < p_card_dict[keys[i]][hand_id].length; j++) {
                console.log("    j: " + j)
                var x = parseInt(player_card_positions[keys[i]][0]) + parseInt(hand_id) * hand_offset;
                var y = parseInt(player_card_positions[keys[i]][1]) + parseInt(j) * card_offset;
                console.log("    AT (" + x + "," + y + ")")
                console.log(p_card_dict[keys[i]][hand_id][j])
                ctx.drawImage(p_card_dict[keys[i]][hand_id][j], x, y, new_card_width, new_card_height);
            }
        }
    }
}

function draw_players_data() {
    var keys = Object.keys(player_data);
    for (var i = 0; i < keys.length; i++){
        console.log(keys[i])
        console.log(player_data_positions[keys[i]])
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


function draw_dealer_cards() {
    for (var i = 0; i < loaded_dealer_cards_list.length; i++) {
        ctx.drawImage(loaded_dealer_cards_list[i], dealer_card_positions[0]+i*card_width_offset, dealer_card_positions[1], new_card_width, new_card_height);
    }
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
    action_list = [];
    var player_amount = Object.keys(ac_dict).length

    for (var p_id = 0; p_id < player_amount; p_id++) {
        for (var j = 0; j < ac_dict[p_id].length; j++) {
            action_list.push(ac_dict[p_id][j]);
        }
    }
    /*
    console.log("ACTION LIST CREATED: ")
    for (var ac_idx = 0; ac_idx < action_list.length; ac_idx++) {
        console.log(action_list[ac_idx])
    }
    */
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
    draw_dealer_cards();
    
    //draw_current_bets();
    //draw_folded();
    //draw_cards_on_table();

    
}

