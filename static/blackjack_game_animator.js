var canvas, ctx, img, fps, interval, globalscale;
var player_card_positions, card_offset, player_data_positions, player_bet_positions, dealer_card_positions, action_text_positions, hand_offset,card_width_offset, player_actions, action_map;
var player_data, cross_img, curr_action_idx, curr_action_subcounter, action_list, loaded_player_card_dict, loaded_dealer_cards_list, dealer_cards_values, game_done, loaded_backside_card, p_card_amount;
var xscale, yscale, og_card_width, og_card_height, new_card_width, new_card_height, bigfont, mediumfont, smallfont, player_data_offset, results, card_count, all_player_cards;
var dealer_counter;
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

    loaded_backside_card = new Image();
    loaded_backside_card.src = "/static/Cards/backside.png";

    cards_on_table = 0;
    dealer_counter = 1
    game_done = false;

    player_actions = ["", "", "", "", "", ""]
    action_map = {"Stand": "S", "Hit": "H", "Split": "SP", "Double": "D"}

    display_cards = [[2], [2], [2], [2], [2], [2]]
    card_count = [[0], [0], [0], [0], [0], [0]]

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
        draw_dealer_cards(1);
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

function load_cards(player_cards, dealer_cards) {
    dealer_cards_values = []
    all_player_cards = {};
    loaded_player_card_dict = {};
    loaded_dealer_cards_list = [];
    var p_keys = Object.keys(player_cards);
    for (var i = 0; i < p_keys.length; i++) {
        var p_id = p_keys[i]
        loaded_player_card_dict[p_id] = {}
        all_player_cards[p_id] = {}
        var hands_dict = player_cards[p_id]
        var hand_amount = Object.keys(hands_dict).length
        for (var hand_id = 0; hand_id < hand_amount; hand_id++) {
            var temp_hand_lst = []
            var temp_val_lst = []
            var cards_on_hand = hands_dict[hand_id].length
            for (var card_id = 0; card_id < cards_on_hand; card_id++) {
                temp_val_lst.push(player_cards[p_id][hand_id][card_id]["current_rank"])
                temp_hand_lst.push(new Image())
                temp_hand_lst[temp_hand_lst.length - 1].src = "/static/Cards/" + get_card_filename([player_cards[p_id][hand_id][card_id]["current_rank"], player_cards[p_id][hand_id][card_id]["current_suit"], player_cards[p_id][hand_id][card_id]["current_value"]])
            }
            loaded_player_card_dict[p_id][hand_id] = temp_hand_lst
            all_player_cards[p_id][hand_id] = temp_val_lst
        }
    }

    for (var i = 0; i < dealer_cards.length; i++) {
        dealer_cards_values.push(dealer_cards[i]["current_rank"]);
        loaded_dealer_cards_list.push(new Image());
        loaded_dealer_cards_list[loaded_dealer_cards_list.length - 1].src = "/static/Cards/" + get_card_filename([dealer_cards[i]["current_rank"], dealer_cards[i]["current_suit"], dealer_cards[i]["current_value"]]);
    }
}

function set_bals(bals_dict) {
    var start_bals = bals_dict["start_bal"]
    var bets = bals_dict["bets"]
    var p_keys = Object.keys(start_bals);
    for (var i = 0; i < p_keys.length; i++) {
        player_data[p_keys[i]] = {"bal": Math.round(((parseFloat(start_bals[p_keys[i]]) - parseFloat(bets[p_keys[i]])) + Number.EPSILON) * 100) / 100, "bet": bets[p_keys[i]]};
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
    results = json_data['result']
    console.log(results)
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
    console.log(card)
    if ((card[2] > 10) || (card[2] == 1)) {
        if (card[2] == 11) { rank_str = "jack"; }
        if (card[2] == 12) { rank_str = "queen"; }
        if (card[2] == 13) { rank_str = "king"; }
        if (card[2] == 1) { rank_str = "ace"; }
    }
    var filename = rank_str + "_of_" + card[1].toString().toLowerCase() + ".png";
    console.log(filename + "\n\n")
    return filename;
}


function draw_player_cards(p_card_dict) {
    var keys = Object.keys(p_card_dict);
    ctx.font = "900 " + bigfont + "px Arial";
    ctx.fillStyle = "red"
    //card_count = [[0], [0], [0], [0], [0], [0]]
    for (var i = 0; i < keys.length; i++) {
        var hand_ids = Object.keys(p_card_dict[keys[i]])
        for (var hand_id = 0; hand_id < display_cards[keys[i]].length; hand_id++) {
            var sum_str = ""
            var sum_int = [0, 0]
            for (var j = 0; j < display_cards[keys[i]][hand_id]; j++) {
                var x = parseInt(player_card_positions[keys[i]][0]) + parseInt(hand_id) * hand_offset;
                var y = parseInt(player_card_positions[keys[i]][1]) + parseInt(j) * card_offset;
                ctx.drawImage(p_card_dict[keys[i]][hand_id][j], x, y, new_card_width, new_card_height);
                var c = parseInt(all_player_cards[keys[i]][hand_id][j])
                if (c == 11) {
                    sum_int[0] += 11
                    sum_int[1] += 1
                } else {
                    sum_int[0] += c
                    sum_int[1] += c
                }
            }
            var offset = 25 * hand_id
            if (sum_int[0] != sum_int[1]) {
                ctx.fillText("" + sum_int[0] + "/" + sum_int[1], player_data_positions[keys[i]][0]+(100 * (globalscale / 100)), player_data_positions[keys[i]][1] + (offset * (globalscale / 100)))
            } else {
                ctx.fillText(sum_int[0].toString(), player_data_positions[keys[i]][0]+(100 * (globalscale / 100)), player_data_positions[keys[i]][1] + (offset * (globalscale / 100)))
            }
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
        ctx.font = "900 " + mediumfont + "px Arial";
        ctx.fillStyle = "red"; 
        ctx.fillText(player_data[keys[i]]["bet"] + " $", player_data_positions[keys[i]][0], player_data_positions[keys[i]][1] + (75 * (globalscale / 100)))
    }
}


function draw_dealer_cards(amount) {
    ctx.fillStyle = "red";
    ctx.font = "900 " + bigfont + "px Arial";
    if (amount == 1) {
        ctx.fillText(dealer_cards_values[0].toString(), dealer_card_positions[0] + (20 * (globalscale / 100)), dealer_card_positions[1] + (150 * (globalscale / 100)))
        var face_up_card = loaded_dealer_cards_list[0]
        ctx.drawImage(face_up_card, dealer_card_positions[0], dealer_card_positions[1], new_card_width, new_card_height);
        ctx.drawImage(loaded_backside_card, dealer_card_positions[0] + card_width_offset, dealer_card_positions[1], new_card_width, new_card_height);
    } else {
        var sum_int = [0, 0]
        for (var i = 0; i < amount; i++) {
            if (i < loaded_dealer_cards_list.length){
                var c = dealer_cards_values[i]
                if (c == 11) {
                    sum_int[0] += 11
                    sum_int[1] += 1
                } else {
                    sum_int[0] += c
                    sum_int[1] += c
                }
                ctx.drawImage(loaded_dealer_cards_list[i], dealer_card_positions[0]+i*card_width_offset, dealer_card_positions[1], new_card_width, new_card_height);
            }
        }
        if (sum_int[0] != sum_int[1]) {
            ctx.fillText(sum_int[0].toString() + "/" + sum_int[1].toString(), dealer_card_positions[0] + (20 * (globalscale / 100)), dealer_card_positions[1]  + (150 * (globalscale / 100)))
        } else {
            ctx.fillText(sum_int[0].toString(), dealer_card_positions[0] + (20 * (globalscale / 100)), dealer_card_positions[1] + (150 * (globalscale / 100)))
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

function draw_player_actions() {
    ctx.fillStyle = "red";
    ctx.font = bigfont + "px Arial";
    for (var i = 0; i < player_actions.length; i++) {
        ctx.fillText(player_actions[i], player_data_positions[i][0] + (10 * (globalscale / 100)), player_data_positions[i][1] - (50 * (globalscale / 100)));
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

function draw_results(){
    keys = Object.keys(results)
    subKeyOffset = 25
    for(var i = 0; i < keys.length; i++){
        subKeys = Object.keys(results[keys[i]])
        for(var j=0; j < subKeys.length; j++){
            ctx.fillText(results[keys[i]][subKeys[j]],player_data_positions[i][0], player_data_positions[i][1] - (100 +(j*subKeyOffset)* (globalscale / 100)))
        }
    }
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    draw_player_cards(loaded_player_card_dict);
    draw_players_data();
    draw_player_actions();
    
    if(curr_action_idx > action_list.length && curr_action_idx <= action_list.length + loaded_dealer_cards_list.length){
        dealer_counter += 1
    } else if (curr_action_idx > action_list.length + loaded_dealer_cards_list.length && curr_action_idx <= action_list.length + loaded_dealer_cards_list.length+5){
        draw_results()
    } else if(curr_action_idx > action_list.length + loaded_dealer_cards_list.length+5){
        draw_results()
        end_game()
    }
    draw_dealer_cards(dealer_counter);

    if(curr_action_idx < action_list.length){
        action = action_list[curr_action_idx];
        print(action, "action")
    
        
    
        if (player_actions[action.player_id].length == 0) {
            player_actions[action.player_id] += action_map[action.action_str]
        } else {
            player_actions[action.player_id] += ", " + action_map[action.action_str]
        }
    
    
        
        if (action.action_str == "Double") {
            display_cards[action.player_id][action.hand_id] += 1
        }
        if (action.action_str == "Hit") {
            display_cards[action.player_id][action.hand_id] += 1
        }
        if (action.action_str == "Split") {
            display_cards[action.player_id].push(2)
        }     
    }
    curr_action_idx += 1 
    //draw_current_bets();
    //draw_folded();
    //draw_cards_on_table();

    
}

