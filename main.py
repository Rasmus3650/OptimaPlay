#from Input.visual import Visual_input
import cv2
import numpy as np
import time
from PIL import ImageGrab
from Input.training import Training
from flask import *
import threading
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

app = Flask(__name__)

def hello():
    return 'you suck'

def run_flask():
    app.run()

def main():
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    print("Web Server Started")
    start_time = time.time()
    number_of_tables=1
    training_obj = Training(number_of_tables)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Total time: {run_time}\nAvg. Time Per Table: {run_time / number_of_tables}")
    #p_hands = [[Card(2, "Hearts"), Card(3, "Spades")], [Card(9, "Clubs"), Card(9, "Diamonds")], [Card(2, "Clubs"), Card(8, "Diamonds")], [Card(11, "Hearts"), Card(10, "Clubs")], [Card(7, "Diamonds"), Card(14, "Diamonds")], [Card(7, "Hearts"), Card(10, "Spades")]]
    #cards_on_table = [Card(5, "Hearts"), Card(9, "Hearts"), Card(7, "Spades"), Card(14, "Clubs"), Card(4, "Diamonds")]
    #hand_eval = Hand_Evaluator()

    #print(hand_eval.compute_hand(p_hands[0], cards_on_table))
    #for p_id in range(len(p_hands)):
    #    res = hand_eval.compute_hand(p_hands[p_id], cards_on_table)
    #    print(f"P {p_id}: {res}")

if __name__ == "__main__":
    main()