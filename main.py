#from Input.visual import Visual_input
import cv2
import numpy as np
import time, os
from PIL import ImageGrab
from Poker.Input.training import PokerTraining
from Blackjack.Input.training import BlackjackTraining
from auxiliary.ConsumerThread import ConsumerThread
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


@app.route('/poker/<table>/replay/<game>')
def get_file(table, game):
    redirect_param = request.args.get('redirect')
    redirect_arg = True if redirect_param and redirect_param.lower() == 'true' else False
    png_file_names = [x for x in list(os.walk("static"))[0][2] if x[-2:] != "js"]
    game_folder = os.path.join(os.path.join(os.path.join(os.getcwd(), "Poker/recorded_tables"), f"{table}"), f"{game}")
    if not os.path.exists(game_folder):
        return redirect("/")
    file_names = ["Actions.csv", "Cards.csv", "InitBals.csv", "log.txt", "metadata.txt", "PostgameBals.csv", "Winners.csv"]
    #json_data = []
    #for file in file_names:
    #    file_path = os.path.join(game_folder, file)
    #    with open(file_path, 'r') as f:
    #        content = f.read()
    #    json_data.append(content)
    json_data = {}
    with open(os.path.join(game_folder, "game_data.json"), 'r') as f:
        json_data = json.load(f)
    fps = request.args.get('fps', default=2, type=int)

    return render_template('replay_game.html',filenames=png_file_names, game_data=json_data, redirect=redirect_arg, fps=fps)


def start_training(verbose=False, tables=1, consumer_thread = None):
    if not verbose:
        sys.stdout = DummyFile()
        sys.stderr = DummyFile()
    start_time = time.time()
    number_of_tables=tables
    training_obj = PokerTraining(number_of_tables, consumer_thread=consumer_thread)
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


    #consumer_thread = ConsumerThread()
    #consumer_thread.start()


    start_training(verbose=False, tables=1, consumer_thread=None)
    #train_blackjack(verbose=True)
    #consumer_thread.join()
if __name__ == "__main__":
    profile_results_file = "optimization_logs/profile_results.prof"
    cProfile.run('main()', profile_results_file)
