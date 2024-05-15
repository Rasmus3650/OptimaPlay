#from Input.visual import Visual_input
import cv2
import numpy as np
import time, os
from PIL import ImageGrab
from Poker.Input.training import PokerTraining
from Blackjack.Input.training import BlackjackTraining
from Backgammon.Input.training import BackgammonTraining
from auxiliary.ConsumerThread import ConsumerThread
from EconSim.test.map_test import map_test
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
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
class DummyFile(object):
    def write(self, x):
        pass

    def flush(self):
        pass


consumer_thread = None

app = Flask(__name__)



def get_games_dict(game_type):
    games_dict = {}
    if not os.path.exists(f"{game_type}/recorded_tables"):
        return games_dict
    tables = sorted(list(os.walk(f"{game_type}/recorded_tables"))[0][1], key=lambda x: int(x.split('_')[1]))

    for table in tables:
        games = sorted(list(os.walk(os.path.join(f"{game_type}/recorded_tables", table)))[0][1], key=lambda x: int(x.split('_')[1]))
        games_dict[table] = games
    return games_dict

@app.context_processor
def inject_data():
    navbar_data = {
        "poker": get_games_dict("Poker"),
        "blackjack": get_games_dict("Blackjack"),
        "backgammon": get_games_dict("Backgammon"),
        "econsim": get_games_dict("EconSim")
    }
    return dict(navbar_data=navbar_data)

@app.route('/')
def index():
    # Navbar

    # CMS over games, modeller, ...?


    return render_template("index.html")

@app.route('/games')
def games_index():
    # Navbar

    # CMS over de forskellige games


    return render_template("games_index.html")


@app.route('/poker')
def poker_index():
    if not os.path.exists("Poker/recorded_tables"):
        os.mkdir("Poker/recorded_tables")
    tables = sorted(list(os.walk("Poker/recorded_tables"))[0][1], key=lambda x: int(x.split('_')[1]))
    games_dict = {}
    for table in tables:
        games = sorted(list(os.walk(os.path.join("Poker/recorded_tables", table)))[0][1], key=lambda x: int(x.split('_')[1]))
        games_dict[table] = games
    return render_template("poker_index.html", tables=tables, games=games_dict)

@app.route('/blackjack')
def blackjack_index():
    if not os.path.exists("Blackjack/recorded_tables"):
        os.mkdir("Blackjack/recorded_tables")
    tables = sorted(list(os.walk("Blackjack/recorded_tables"))[0][1], key=lambda x: int(x.split('_')[1]))
    games_dict = {}
    for table in tables:
        games = sorted(list(os.walk(os.path.join("Blackjack/recorded_tables", table)))[0][1], key=lambda x: int(x.split('_')[1]))
        games_dict[table] = games
    return render_template("blackjack_index.html", tables=tables, games=games_dict)

@app.route('/backgammon')
def backgammon_index():
    if not os.path.exists("Backgammon/recorded_tables"):
        os.mkdir("Backgammon/recorded_tables")
    tables = sorted(list(os.walk("Backgammon/recorded_tables"))[0][1], key=lambda x: int(x.split('_')[1]))
    games_dict = {}
    for table in tables:
        games = sorted(list(os.walk(os.path.join("Backgammon/recorded_tables", table)))[0][1], key=lambda x: int(x.split('_')[1]))
        games_dict[table] = games
    return render_template("backgammon_index.html", tables=tables, games=games_dict)

@app.route('/econsim')
def econsim_index():
    if not os.path.exists("EconSim/recorded_tables"):
        os.mkdir("EconSim/recorded_tables")
    tables = sorted(list(os.walk("EconSim/recorded_tables"))[0][1], key=lambda x: int(x.split('_')[1]))
    games_dict = {}
    for table in tables:
        games = sorted(list(os.walk(os.path.join("EconSim/recorded_tables", table)))[0][1], key=lambda x: int(x.split('_')[1]))
        games_dict[table] = games
    return render_template("econsim_index.html", tables=tables, games=games_dict)

@app.route('/poker/<table>')
def poker_table_index(table):
    return render_template("table_index.html", table=table)

@app.route('/blackjack/<table>')
def blackjack_table_index(table):
    return render_template("table_index.html", table=table)

@app.route('/backgammon/<table>')
def backgammon_table_index(table):
    return render_template("table_index.html", table=table)

@app.route('/econsim/<table>')
def econsim_table_index(table):
    redirect_param = request.args.get('redirect')
    redirect_arg = True if redirect_param and redirect_param.lower() == 'true' else False
    json_data = {}
    print(os.getcwd())
    with open(os.path.join(os.getcwd(), "EconSim/recorded_tables/table_1/Game_0/game_data.json"), 'r') as f:
        json_data = json.load(f)
    return render_template("replay_econsim_game.html", game_data = json_data, redirect=redirect_arg)

@app.route('/poker/<table>/replay/<game>')
def replay_poker(table, game):
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

    return render_template('replay_poker_game.html',filenames=png_file_names, game_data=json_data, redirect=redirect_arg, fps=fps)

@app.route('/blackjack/<table>/replay/<game>')
def replay_blackjack(table, game):
    redirect_param = request.args.get('redirect')
    redirect_arg = True if redirect_param and redirect_param.lower() == 'true' else False
    png_file_names = [x for x in list(os.walk("static"))[0][2] if x[-2:] != "js"]
    game_folder = os.path.join(os.path.join(os.path.join(os.getcwd(), "Blackjack\\recorded_tables"), f"{table}"), f"{game}")
    if not os.path.exists(game_folder):
        return redirect("/")
    json_data = {}
    with open(os.path.join(game_folder, "game_data.json"), 'r') as f:
        json_data = json.load(f)
    fps = request.args.get('fps', default=2, type=int)

    return render_template('replay_blackjack_game.html',filenames=png_file_names, game_data=json_data, redirect=redirect_arg, fps=fps)

@app.route('/backgammon/<table>/replay/<game>')
def replay_backgammon(table, game):
    redirect_param = request.args.get('redirect')
    redirect_arg = True if redirect_param and redirect_param.lower() == 'true' else False
    png_file_names = [x for x in list(os.walk("static"))[0][2] if x[-2:] != "js"]
    game_folder = os.path.join(os.path.join(os.path.join(os.getcwd(), "Backgammon\\recorded_tables"), f"{table}"), f"{game}")
    if not os.path.exists(game_folder):
        return redirect("/")
    json_data = {}
    with open(os.path.join(game_folder, "game_data.json"), 'r') as f:
        json_data = json.load(f)
    fps = request.args.get('fps', default=2, type=int)

    return render_template('replay_backgammon_game.html',filenames=png_file_names, game_data=json_data, redirect=redirect_arg, fps=fps)

@app.route('/poker/start_training', methods=["POST"])
def start_training():
    global consumer_thread
    start_time = time.time()
    number_of_tables=int(request.form['table_n'])
    strategies = []
    for item in list(request.form.keys()):
        if "strategies" in item:
            strategies.append(request.form[item])
    PokerTraining(number_of_tables, consumer_thread=consumer_thread, strategies=strategies)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Total time: {run_time}\nAvg. Time Per Table: {run_time / number_of_tables}")
    return redirect("/poker")

@app.route('/blackjack/start_training', methods=["POST"])
def train_blackjack():
    global consumer_thread

    start_time = time.time()
    strategies = []
    number_of_tables=int(request.form['table_n'])
    for item in list(request.form.keys()):
        if "strategies" in item:
            strategies.append(request.form[item])
    BlackjackTraining(number_of_tables, consumer_thread, strategies=strategies)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Total time: {run_time}\nAvg. Time Per Table: {run_time / number_of_tables}")
    return redirect("/blackjack")

@app.route('/backgammon/start_training', methods=["POST"])
def train_backgammon():
    global consumer_thread 
    start_time = time.time()
    strategies = []
    number_of_tables=int(request.form['table_n'])
    for item in list(request.form.keys()):
        if "strategies" in item:
            strategies.append(request.form[item])
    BackgammonTraining(number_of_tables, consumer_thread, strategies=strategies)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Total time: {run_time}\nAvg. Time Per Table: {run_time / number_of_tables}")
    return redirect("/backgammon")





def main():

    consumer_thread = ConsumerThread()
    consumer_thread.start()
    # start_training(tables=5, consumer_thread=consumer_thread)
    # train_blackjack(consumer_thread, verbose=True)
    # train_backgammon(consumer_thread, verbose=True)
    map_test(consumer_thread)
    consumer_thread.stop()
    consumer_thread.join()

def hello_word():
    print("hello")
    
def app_main():
    global consumer_thread
    consumer_thread = ConsumerThread()
    consumer_thread.start()
    app.run()
    consumer_thread.stop()
    consumer_thread.join()

if __name__ == "__main__":
    profile_results_file = "optimization_logs/profile_results.prof"
    #cProfile.run('main()', profile_results_file)
    cProfile.run('app_main()', profile_results_file)
    
