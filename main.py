#from Input.visual import Visual_input
import cv2
import numpy as np
import time, os
from PIL import ImageGrab
from Input.training import Training
from flask import *
import threading
import sys
from game_logic.card import Card
from game_logic.hand_evaluator import Hand_Evaluator

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

@app.route('/')
def index():
    tables = sorted(list(os.walk("recorded_tables"))[0][1], key=lambda x: int(x.split('_')[1]))
    games_dict = {}
    for table in tables:
        games = sorted(list(os.walk(os.path.join("recorded_tables", table)))[0][1], key=lambda x: int(x.split('_')[1]))
        games_dict[table] = games
    return render_template("index.html", tables=tables, games=games_dict)


@app.route('/replay/<table>/<game>')
def get_file(table, game):
    png_file_names = [x for x in list(os.walk("static"))[0][2] if x[-2:] != "js"]
    game_folder = os.path.join(os.path.join(os.path.join(os.getcwd(), "recorded_tables"), f"{table}"), f"{game}")
    json_data = []
    for root, dirs, files in os.walk(game_folder):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                content = f.read()
                
            json_data.append(content.replace("\n", ""))
    return render_template('replay_game.html', filenames=png_file_names, game_data=json_data)


def start_training(verbose=False):
    if not verbose:
        sys.stdout = DummyFile()
        sys.stderr = DummyFile()
    start_time = time.time()
    number_of_tables=1
    training_obj = Training(number_of_tables)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Total time: {run_time}\nAvg. Time Per Table: {run_time / number_of_tables}")

def train():
    train_thread = threading.Thread(target=start_training, args=(False,))
    train_thread.start()
    train_thread.join()

def main():
    #for _ in range(3):
    #    train()
    print("Web Server Started")
    app.run()

    

if __name__ == "__main__":
    main()