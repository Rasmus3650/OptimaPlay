import cv2
import numpy as np
import pytesseract
import time
from PIL import ImageGrab
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.card import Card
from game_logic.table import Table

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class Visual_input():
    def __init__(self, number_of_tables: int = 1):
        self.table_list = [Table(1.6, side=0), Table(1.6, side=1)]
        self.width = 120
        self.height = 37

        self.number_width = 18
        self.number_height = 22
        self.number_offset = 60

        self.suit_width = 18
        self.suit_height = 10
        self.suit_offset = 22

        id0 = [[52, 422], [52, 1382]]   #Done
        id1 = [[115, 750], [115, 1710]] #Done
        id2 = [[318, 77], [318, 1739]] #Done
        id3 = [[437, 422], [437, 1382]] #Done
        id4 = [[318, 64], [318, 1024]]  #Done? Line on the right card
        id5 = [[116, 94], [116, 1054]]  #Done
        
        self.card_regions = [id0, id1, id2, id3, id4, id5]

        self.card_rank_map = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

        self.monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        self.monitor_tuple = (0, 0, 1920, 1080)

        self.spades_threshold = [51, 144]
        self.hearts_threshold = [51, 144]
        self.clubs_threshold = [0, 50]
        self.diamonds_threshold = [0, 50]
        

        self.red_thresholds = [self.hearts_threshold, self.diamonds_threshold]
        self.black_thresholds = [self.spades_threshold, self.clubs_threshold]
        self.red_suits = ["Hearts", "Diamonds"]
        self.black_suits = ["Spades", "Clubs"]
        
        self.all_game_states = ["Pre-round", "Pre-flop", "Flop", "Turn", "River", "Showdown", "Conclusion"]


    def get_rank(self, rank_img, x):
        gray = cv2.cvtColor(rank_img, cv2.COLOR_BGR2GRAY)
        # gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # gray = cv2.bitwise_not(cv2.bitwise_not(img_bin))
        cv2.imwrite(f"./assets/temp{x}.jpg", gray)
        rec_str = pytesseract.image_to_string(gray, config='--psm 10').strip().upper()
        for key in list(self.card_rank_map.keys()):
            if key in rec_str:
                return self.card_rank_map[key]
        return None
    

    def get_suit(self, image):
        suit_img = np.copy(image)
        
        color = self.get_color(suit_img)
        
        
        gray = cv2.cvtColor(suit_img, cv2.COLOR_BGR2GRAY)
        
        for row in range(len(gray)):
            for col in range(len(gray[row])):
                if gray[row][col] < 150:
                    gray[row][col] = 1
                else:
                    gray[row][col] = 0
        print(gray)
        #print(f"Color: {color}")
        card_sum = sum(sum(gray))
        print(f"Card_sum: {card_sum}")
        if color == "Red":
            for threshold, suit in zip(self.red_thresholds, self.red_suits):
                if card_sum > threshold[0] and card_sum < threshold[1]:
                    return suit
        elif color == "Black":
            for threshold, suit in zip(self.black_thresholds, self.black_suits):
                if card_sum > threshold[0] and card_sum < threshold[1]:
                    return suit
        else:
            print(f"Fucked color: {color}")
            return None

    def get_color(self, card_suit):
        #print(card_suit[5, 8])
        if card_suit[5, 8, 2] > 150:
            return "Red"
        else:
            return "Black"
        

    def get_image(self, image=None):
        if image is not None: return image
        image = np.array(ImageGrab.Grab(bbox=self.monitor_tuple))
        return image
    
    def get_card_at_coordinates(self, y, x, image):
        card_rank = image[y : y + self.number_height,
                               x:x + self.number_width]
        
        card_suit = image[y + self.suit_offset  : y + self.suit_offset + self.suit_height,
                               x : x + self.suit_width]
        rank = self.get_rank(card_rank, x)
        suit = self.get_suit(card_suit)
        
        return Card(rank, suit)
        
    
    
    def get_cards_on_hand(self, curr_pos, curr_id, image):
        
        y = self.card_regions[curr_id][curr_pos][0]
        x = self.card_regions[curr_id][curr_pos][1]

        left_card = self.get_card_at_coordinates(y, x, image)
        right_card = self.get_card_at_coordinates(y, x + self.number_offset, image)
        print(left_card)
        print(right_card)
        
        return [left_card, right_card]


    def get_number_of_cards(self, init_image):
        image = np.copy(init_image)
        n = 0
        for table in self.table_list:
            for point in table.corner_points:
                print(f"Corner Point: {image[point[0], point[1]]}")
                if np.all(image[point[0], point[1]]) >= 200:
                    n += 1
                else:
                    break
        return n

    def check_game_state(self, game_state: str, amount_on_table: int):
        if amount_on_table == 0:
            return game_state == "Pre-Round" or game_state == "Pre-Flop"
        if amount_on_table == 3:
            return game_state == "Flop"
        if amount_on_table == 4:
            return game_state == "Turn"
        if amount_on_table == 5:
            return game_state == "River" or game_state == "Showdown" or game_state == "Conclusion"

        
    def get_cards_on_table(self, game_states: list[str], image):
        cards_on_tables = [[], []] 
        for table in self.table_list:
            amount = self.get_number_of_cards(image)
            for point in table.corner_points[:amount]:
                new_card = self.get_card_at_coordinates(point[0], point[1], image)
                if table.side == 1:
                    print(point)
                table.deck.draw_card(new_card)
                cards_on_tables[table.side].append(new_card)
        return cards_on_tables        
    
    def print_area(self, y, x, init_image, radius = 10, filename = None):
        image = [np.copy(img_row) for img_row in init_image]
        image = np.array(image)
        print(f"Making coutout, dims: ({y}, {x})")
        
        print(f"Value at y: {y}, x: {x} = {image[y, x]}")
        image[y, x] = [0, 0, 0]
        

        cut_out = image[y - radius : y + radius,
                        x - radius : x + radius]
        
        if filename is not None:
            cv2.imwrite(f"./assets/{filename}", cut_out)
        else:
            cv2.imshow(f"Print area at y = {y}, x = {x}, r = {radius}", cut_out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
