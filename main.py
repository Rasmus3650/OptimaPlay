#from Input.visual import Visual_input
import cv2
import numpy as np
import time, os
from PIL import ImageGrab
from Poker.Input.training import PokerTraining
from Blackjack.Input.training import BlackjackTraining
from flask import *
import threading
import sys
from Poker.game_logic.card import Card
from Poker.game_logic.hand_evaluator import Hand_Evaluator
import plotly
import plotly.graph_objs as go
import pstats
import cProfile
import snakeviz
# def main2():
#     vis = Visual_input()
#     image = cv2.imread("./assets/img1.jpg")
#     id = 4
#     side = 1
#     cards = vis.get_cards_on_hand(side, id, image)
#     #image = None
#     #print(f"CARDS ON HAND:")
#     #for card in cards: print(f"  {card}")

#     print(f"Calling get_cards_on_table")
#     on_table = vis.get_cards_on_table(["Pre-Flop","Turn"], image)

#     for i, side in enumerate(["Left", "Right"]):
#         print(f"{side} table:")
#         for card in on_table[i]:
#             print(f"  {card.current_rank}, {card.current_suit}")
#         print()
    
    
#     #vis.get_cards_on_table("River", image, side=0)
class DummyFile(object):
    def write(self, x):
        pass

    def flush(self):
        pass


app = Flask(__name__)

@app.context_processor
def inject_data():
    navbar_data = {"poker": {}, "blackjack": {}}

    tables = sorted(list(os.walk("Poker/recorded_tables"))[0][1], key=lambda x: int(x.split('_')[1]))
    games_dict = {}
    for table in tables:
        games = sorted(list(os.walk(os.path.join("Poker/recorded_tables", table)))[0][1], key=lambda x: int(x.split('_')[1]))
        games_dict[table] = games
    navbar_data['poker'] = games_dict
    return dict(navbar_data=navbar_data)

@app.route('/')
def index():
    # Navbar

    # CMS over games, modeller, ...?


    return render_template("index.html")
@app.route('/navbar')
def nav():
    return render_template("navbar.html")

@app.route('/games')
def games_index():
    # Navbar

    # CMS over de forskellige games


    return render_template("games_index.html")


@app.route('/poker')
def poker_index():
    tables = sorted(list(os.walk("Poker/recorded_tables"))[0][1], key=lambda x: int(x.split('_')[1]))
    games_dict = {}
    for table in tables:
        games = sorted(list(os.walk(os.path.join("Poker/recorded_tables", table)))[0][1], key=lambda x: int(x.split('_')[1]))
        games_dict[table] = games
    return render_template("poker_index.html", tables=tables, games=games_dict)

@app.route('/blackjack')
def blackjack_index():
    
    return render_template("blackjack_index.html")

@app.route('/poker/<table>')
def table_index(table):


    return render_template("table_index.html", table=table)

def get_bals(table, game, initbals=False):
    bal_str = ""
    game_folder = os.path.join(os.path.join(os.path.join(os.getcwd(), "Poker/recorded_tables"), f"{table}"), f"{game}")
    if initbals:
        file_path = os.path.join(game_folder, "InitBals.csv")
    else:
        file_path = os.path.join(game_folder, "PostgameBals.csv")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            bal_str = f.read()
    bal_dict = {}
    players, bals = bal_str.split("\n")
    players = players.split(", ")
    bals = bals.split(", ")
    for i in range(len(players)):
        p_id = int(players[i][-1])
        bal_dict[p_id] = float(bals[i])
    #print(bal_dict)
    return bal_dict

@app.route('/poker/<table>/replay/<game>')
def get_file(table, game):
    redirect_param = request.args.get('redirect')
    redirect_arg = True if redirect_param and redirect_param.lower() == 'true' else False
    png_file_names = [x for x in list(os.walk("static"))[0][2] if x[-2:] != "js"]
    game_folder = os.path.join(os.path.join(os.path.join(os.getcwd(), "Poker/recorded_tables"), f"{table}"), f"{game}")
    if not os.path.exists(game_folder):
        return redirect("/")
    file_names = ["Actions.csv", "Cards.csv", "InitBals.csv", "log.txt", "metadata.txt", "PostgameBals.csv", "Winners.csv"]
    json_data = []
    for file in file_names:
        file_path = os.path.join(game_folder, file)
        with open(file_path, 'r') as f:
            content = f.read()
        json_data.append(content)
    fps = request.args.get('fps', default=2, type=int)

    # ----- NEDENUNDER ER TIL AT PLOTTE PLAYER BAL HENOVER TID
 
    bal_history = [get_bals(table, f"Game_0", initbals=True)]
    for i in range(int(game.split("_")[1])):
        bal_history.append(get_bals(table, f"Game_{i}", initbals=False))
    bal_data = []

    for player_id in range(0,6):
        y_arr = []
        for game, balances in enumerate(bal_history):
            balance = balances.get(player_id, None)
            if balance is not None:
                y_arr.append(balance)
            else:
                y_arr.append(0)  # or any default value you prefer
            
        trace = go.Scatter(
            x=list(range(len(bal_history))),
            y=y_arr,
            mode='lines+markers',
            name=f'Player {player_id}'
        )
        bal_data.append(trace)


    layout = go.Layout(
        title='Player Balances Over Time',
        xaxis=dict(title='Game Index'),
        yaxis=dict(title='Balance')
    )
    fig = go.Figure(data=bal_data, layout=layout)

    plot_json = json.dumps(fig.to_dict())
    print(f"Sanity check")
    return render_template('replay_game.html', plot_json=plot_json,filenames=png_file_names, game_data=json_data, redirect=redirect_arg, fps=fps)


def start_training(verbose=False):
    if not verbose:
        sys.stdout = DummyFile()
        sys.stderr = DummyFile()
    start_time = time.time()
    number_of_tables=10
    training_obj = PokerTraining(number_of_tables)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Total time: {run_time}\nAvg. Time Per Table: {run_time / number_of_tables}")

def train():
    train_thread = threading.Thread(target=start_training, args=(True,))
    train_thread.start()
    train_thread.join()

def train_blackjack(verbose=False):
    if not verbose:
        sys.stdout = DummyFile()
        sys.stderr = DummyFile()
    start_time = time.time()
    number_of_tables=1
    training_obj = BlackjackTraining(number_of_tables)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Total time: {run_time}\nAvg. Time Per Table: {run_time / number_of_tables}")

def main():
    #for _ in range(5):
    #    train()
    #print("Web Server Started")
    #app.run()
    start_training(verbose=False)
    #train_blackjack(verbose=True)

if __name__ == "__main__":
    profile_results_file = "optimization_logs/profile_results.prof"
    cProfile.run('main()', profile_results_file)
