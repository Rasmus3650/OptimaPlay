var canvas, ctx, img, fps, interval, actions, cards, initbals, postgamebals, winners, metadata, log, player_card_positions, card_offset, dealer_chip_positions, player_data_positions, dealer_chip_img, player_bet_positions, table_card_positions, pot_pos, pot, action_text_positions, player_data, cross_img, curr_action_idx, curr_action_subcounter, action_list, loaded_player_card_dict, loaded_table_cards_list, cards_on_table, dealer, game_done;
window.onload = (event) => {
    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext("2d");

    dealer_chip_img = new Image();
    dealer_chip_img.src = "/static/Dealer_brick.png";

    img = new Image();
    img.src = "/static/Poker_table.png";
    ctx.drawImage(img, 0, 0);

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
        clearInterval(interval);
        interval = setInterval(()=>{//
            animate();
        },1000/fps);
    })

    loaded_player_card_dict = {};
    loaded_table_cards_list = [];

    curr_action_idx = 0;
    curr_action_subcounter = 0;

    player_card_positions = {0: [720, 140], 1: [1180, 190], 2: [1180, 510], 3: [720, 560], 4: [260, 510], 5: [260, 190]}
    dealer_chip_positions = {0: [670, 140], 1: [1200, 150], 2: [1350, 510], 3: [885, 565], 4: [215, 520], 5: [215, 220]}
    player_data_positions = {0: [770, 30], 1: [1380, 85], 2: [1460, 660], 3: [770, 735], 4: [90, 660], 5: [160, 85]}
    player_bet_positions = {0: [790, 280], 1: [1165, 320], 2: [1165, 505], 3: [790, 555], 4: [410, 505], 5: [410, 320]}
    var table_cards_y = 360;
    table_card_positions = {0: [602, table_cards_y], 1: [682, table_cards_y], 2: [762, table_cards_y], 3: [842, table_cards_y], 4: [922, table_cards_y]}
    card_offset = 80;

    action_text_positions = {0: [745, 90], 1: [1355, 145], 2: [1440, 720], 3: [740, 795], 4: [60, 720], 5: [125, 145]};

    pot_pos = [740, 350];
    pot = 0;

    player_data = {0: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   1: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   2: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   3: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   4: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0},
                   5: {"folded": false, "all-in": false, "bal": 0.0, "curr_bet": 0.0, "total_bet": 0.0}};

    load_data();
    start_animation(fps);
};

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

function get_action_dict(actions) {
    actions = actions.split("\\n");
    actions = actions.slice(0, actions.length - 1);
    actions[0] = actions[0].slice(2);
    for (var i = 0; i < actions.length; i++) {
        if (actions[i].length > 0) {
            actions[i] = actions[i].split(",");
        }
    }
    var action_dict = {};
    var done = false;
    for (var i = 0; i < actions[0].length; i++) {
        actions[0][i] = actions[0][i].trim();
        action_dict[actions[0][i]] = [];
        for (var j = 1; j < actions.length; j++) {
            if (actions[j][i].length < 3) {
                done = true;
                break
            }
            actions[j][i] = actions[j][i].slice(1, actions[j][i].length - 1).split(";");
            var p_id = parseInt(actions[j][i][0]);
            var ac_str = actions[j][i][1];
            var amount = parseFloat(actions[j][i][2]);
            action_dict[actions[0][i]].push([p_id, ac_str, amount]);
        }
        if (done) {
            break;
        }
    }
    var keys = Object.keys(action_dict);
    for (var i = 0; i < keys.length; i++) {
        if (action_dict[keys[i]].length == 0) {
            delete action_dict[keys[i]];
        }
    }
    return action_dict;
}

function get_card_dict_and_table_cards(cards) {
    var card_dict = {};
    var table_cards = [];
    console.log(cards)
    cards = cards.split("\\n");
    for (var i = 0; i < cards.length; i++) {
        if (cards[i].length != 0 && cards[i].length < 70) {
            var id = cards[i][2];
            var card1 = cards[i].slice(cards[i].indexOf("d(") + 2, cards[i].indexOf("),")).split(",");
            var card2 = cards[i].slice(cards[i].indexOf("),") + 8, cards[i].length - 2).split(",");
            
            card_dict[parseInt(id)] = [[parseInt(card1[0]), card1[1].slice(2, card1[1].length - 1)], [parseInt(card2[0]), card2[1].slice(2, card2[1].length - 1)]]

        } else if (cards[i].length >= 70) {
            for (var j = 0; j < 5; j++) {
                var card = cards[i].slice(cards[i].indexOf("d(") + 2, cards[i].indexOf("),"));
                if (j == 4) {
                    card = card.slice(0, card.length - 1);
                }
                card = card.split(",");
                table_cards.push([parseInt(card[0]), card[1].slice(2, card[1].length - 1)])
                cards[i] = cards[i].slice(cards[i].indexOf("),") + 1);
            }
        }
    }
    return [card_dict, table_cards];
}

function get_bals_dict(bal_str) {
    bal_dict = {};
    console.log(bal_str)
    var [players, bals] = bal_str.split("\\n");
    players = players.split(", ");
    bals = bals.split(", ");
    for (var i = 0; i < players.length; i++) {
        var p_id = parseInt(players[i].slice(players[i].length - 1));
        bal_dict[p_id] = parseFloat(bals[i]);
    }
    return bal_dict;
}

function get_dealer(metadata) {
    return parseInt(metadata.replace("Dealer: ", ""));
}

function get_winner(winners) {
    return JSON.parse(winners.replace(/Player\s(\d+)/g, '$1').replace("']", ""));
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
    let data = document.getElementById("game_data").innerText;
    [actions, metadata, initbals, winners, cards, postgamebals, log] = data.split("', '");
    console.log("ACTIONS: ", actions)
    console.log("CARDS: ", cards)
    console.log("INITBALS: ", initbals)
    console.log("LOG: ", log)
    console.log("METADATA: ", metadata)
    console.log("POSTGAMEBALS: ", postgamebals)
    console.log("WINNERS: ", winners)
    var action_dict = get_action_dict(actions);
    var [card_dict, table_cards] = get_card_dict_and_table_cards(cards);
    load_cards(card_dict, table_cards);
    var initbals_dict = get_bals_dict(initbals);
    set_bals(initbals_dict);
    dealer = get_dealer(metadata);
    var postgamebals_dict = get_bals_dict(postgamebals);
    var winner_arr = get_winner(winners);
    
    print(action_dict, "Action dict");
    print(card_dict, "Card dict");
    print(table_cards, "Table cards");
    print(initbals_dict, "Init bals");
    print(dealer, "Dealer");
    print(postgamebals_dict, "Postgame bals");
    print(winner_arr, "Winners");

    create_action_list(action_dict);
    console.log(action_list);
    
    

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
}

function get_card_filename(card) {
    var rank_str = card[0].toString();
    if (card[0] > 10) {
        if (card[0] == 11) { rank_str = "jack"; }
        if (card[0] == 12) { rank_str = "queen"; }
        if (card[0] == 13) { rank_str = "king"; }
        if (card[0] == 14) { rank_str = "ace"; }
    }
    var filename = rank_str + "_of_" + card[1].toLowerCase() + ".png";
    return filename;
}


function draw_player_cards(p_card_dict) {
    var keys = Object.keys(p_card_dict);
    for (var i = 0; i < keys.length; i++) {
        for (var j = 0; j < p_card_dict[keys[i]].length; j++) {
            var x = parseInt(player_card_positions[keys[i]][0]) + parseInt(j) * card_offset;
            var y = parseInt(player_card_positions[keys[i]][1]);

            ctx.drawImage(p_card_dict[keys[i]][j], x, y);
        }
    }
}

function draw_players_data() {
    var keys = Object.keys(player_data);
    for (var i = 0; i < keys.length; i++){
        var x = parseInt(player_data_positions[keys[i]][0])
        var y = parseInt(player_data_positions[keys[i]][1]+5)
        ctx.font = "32px Arial";
        ctx.fillStyle = "white"; 
        ctx.fillText("P "+keys[i],x,y)
        ctx.font = "20px Arial";
        var x_bal = parseInt(player_data_positions[keys[i]][0])
        var y_bal = parseInt(player_data_positions[keys[i]][1]+25)
        ctx.fillText(player_data[keys[i]]["bal"] + " $", x_bal, y_bal)
    }
}


function draw_dealer_chip() {
    ctx.drawImage(dealer_chip_img, dealer_chip_positions[dealer][0], dealer_chip_positions[dealer][1]);
}

function draw_cards_on_table() {
    for (var i = 0; i < cards_on_table; i++) {
        ctx.drawImage(loaded_table_cards_list[i], table_card_positions[i][0], table_card_positions[i][1]);
    }
}

function draw_pot() {
    ctx.font = "32px Arial";
    ctx.fillStyle = "white"; 
    ctx.fillText("Pot: " + pot + " $", pot_pos[0], pot_pos[1]);
}

function draw_current_bets() {
    var p_ids = Object.keys(player_data);
    ctx.font = "18px Arial";
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
    ctx.font = "bold 18px Arial";
    ctx.fillStyle = "green"; 
    ctx.fillText(ac_str, action_text_positions[action[0]][0], action_text_positions[action[0]][1])
}

function draw_cross(player_id){
    ctx.drawImage(cross_img, player_card_positions[player_id][0], player_card_positions[player_id][1])
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
        let result = baseUrl + "Game_" + game + "?redirect=True&fps="+fps;
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
    ctx.drawImage(img, 0, 0);
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

