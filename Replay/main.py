import sys, os, csv, time
from tkinter import filedialog
import pygame
from pygame.locals import *
from pathlib import Path
import re

class Action():
    def __init__(self, player_id, action_str, amount) -> None:
        self.player_id = player_id
        self.action_str = action_str
        self.amount = amount
    
    def __repr__(self) -> str:
        return f"P{self.player_id}: {self.action_str} ({self.amount} $)"
    


pygame.init()

font = pygame.font.Font('freesansbold.ttf', 32)
underline_font = pygame.font.Font('freesansbold.ttf', 32)
underline_font.set_underline(True)
small_font = pygame.font.Font('freesansbold.ttf', 16)
player_positions = {0: (770, 55), 1: (1380, 110), 2: (1460, 685), 3: (770, 760), 4: (90, 685), 5: (160, 110)}
card_positions = {0: (720, 190), 1: (1180, 240), 2: (1180, 560), 3: (720, 610), 4: (260, 560), 5: (260, 240)}
card_x_offset = 80
dealer_brick_positions = {0: (670, 190), 1: (1200, 200), 2: (1350, 560), 3: (885, 615), 4: (215, 570), 5: (215, 270)}
action_text_pos = {0: (745, 20), 1: (1355, 80), 2: (1440, 655), 3: (740, 880), 4: (60, 655), 5: (125, 80)}
player_bet_amount = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
player_bet_positions = {0: (790, 315), 1: (1165, 355), 2: (1165, 540), 3: (790, 590), 4: (410, 540), 5: (410, 355)}
pot_pos = (700, 365)
game_number_pos = (300, 50)
folded_players = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}
#fps = 60
fps = 1
fpsClock = pygame.time.Clock()

width, height = 1600, 900
screen = pygame.display.set_mode((width, height))


def get_bals(game_folder_path, init=True):
    if init:
        bals_csv = open(os.path.join(game_folder_path, "InitBals.csv"))
    else:
        bals_csv = open(os.path.join(game_folder_path, "PostgameBals.csv"))
    players = bals_csv.readline().replace("\n", "").replace(" ", "").replace("P", "").split(",")
    bals = bals_csv.readline().replace("\n", "").replace(" ", "").split(",")
    player_dict = {int(k):float(v) for k, v in zip(players, bals)}
    return player_dict


def place_players(screen, player_bal_dict, curr_player = None):
    white = (255, 255, 255)
    green = (0, 255, 0)
    for player_id in list(player_bal_dict.keys()):
        if player_id not in list(player_positions.keys()): break
        if curr_player is not None and player_id == curr_player:
            screen.blit(underline_font.render(f"P {player_id}", True, green), player_positions[player_id])
        else:
            screen.blit(underline_font.render(f"P {player_id}", True, white), player_positions[player_id])
        screen.blit(small_font.render(str(player_bal_dict[player_id]) + " $", True, white), (player_positions[player_id][0], player_positions[player_id][1] + 40))

def get_cards(game_folder_path):
    cards_csv = open(os.path.join(game_folder_path, "Cards.csv"))
    lines = cards_csv.readlines()
    cards_csv.close()
    curr_line = 0
    card_pattern = '[0-9]*,\s"[a-zA-Z]*"'
    player_cards = {}
    card_map = {11: "jack", 12: "queen", 13: "king", 14: "ace"}
    table_cards = []

    while lines[curr_line][0] == "P":
        line = lines[curr_line]
        p_id = int(line[2])
        res = [x.replace('"', '').replace(" ", "").split(",") for x in re.findall(card_pattern, line)]
        c1 = res[0]
        c2 = res[1]
        c1_str = f""
        c2_str = f""
        if int(c1[0]) > 10:
            c1_str += card_map[int(c1[0])]
        else:
            c1_str += c1[0]

        if int(c2[0]) > 10:
            c2_str += card_map[int(c2[0])]
        else:
            c2_str += c2[0]

        c1[1] = c1[1].lower()
        c2[1] = c2[1].lower()
        player_cards[p_id] = [[int(c1[0]), c1[1], f"{c1_str}_of_{c1[1]}.png"], [int(c2[0]), c2[1], f"{c2_str}_of_{c2[1]}.png"]]
        curr_line += 1
    
    while lines[curr_line][0] != "[":
        curr_line += 1
    
    res = [x.replace('"', '').replace(" ", "").split(",") for x in re.findall(card_pattern, lines[curr_line])]
    for c in res:
        c[1] = c[1].lower()
        c_str = f""
        if int(c[0]) > 10:
            c_str += card_map[int(c[0])]
        else:
            c_str += c[0]
        table_cards.append([int(c[0]), c[1], f"{c_str}_of_{c[1]}.png"])

    return player_cards, table_cards

def load_cards(player_cards, table_cards, assets_folder="assets/Cards"):
    for p_id in list(player_cards.keys()):
        player_cards[p_id][0].append(pygame.image.load(os.path.join(assets_folder, player_cards[p_id][0][2])))
        player_cards[p_id][1].append(pygame.image.load(os.path.join(assets_folder, player_cards[p_id][1][2])))
    
    for c in table_cards:
        c.append(pygame.image.load(os.path.join(assets_folder, c[2])))

    return player_cards, table_cards

def place_player_cards(screen, player_cards):
    for p_id in list(player_cards.keys()):
        if p_id not in list(card_positions.keys()): break
        pos = card_positions[p_id]
        screen.blit(player_cards[p_id][0][3], pos)
        screen.blit(player_cards[p_id][1][3], (pos[0] + card_x_offset, pos[1]))

def place_dealer_brick(screen, dealer, dealer_brick):
    if dealer not in list(dealer_brick_positions.keys()): return
    screen.blit(dealer_brick, dealer_brick_positions[dealer])

        
def get_dealer(game_folder_path):
    metadata_file = open(os.path.join(game_folder_path, "metadata.txt"))
    lines = metadata_file.readlines()
    metadata_file.close()
    for line in lines:
        if line[0:7] == "Dealer:":
            return int(line[8])

def get_action_str(action):
    ac_str = f"{action.action_str}"
    if action.action_str == "Call" or action.action_str == "Raise":
        ac_str += f" {action_to_perform.amount} $"
    return ac_str

def get_actions(game_folder_path):
    state_map = {0: "Pre-flop", 1: "Flop", 2: "Turn", 3: "River"}
    actions = {}
    with open(os.path.join(game_folder_path, "Actions.csv")) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for state in row:
                    state = state.replace(" ", "")
                    actions[state] = []
                line_count += 1
            else:

                for i in range(4):
                    if len(row[i]) > 2:
                        p_id = int(row[i][1])
                        l_sem_indx = row[i].rfind(";")
                        ac_str = row[i][3:l_sem_indx]
                        am = float(row[i][l_sem_indx + 1:-1])
                        actions[state_map[i]].append(Action(p_id, ac_str, am))
                line_count += 1
    return actions

def place_action_text(screen, player_id, text):
    screen.blit(small_font.render(f"{text}", True, (255, 255, 255)), action_text_pos[player_id])

def place_bet_amount(screen, player_bet_amount):
    for p_id in list(player_bet_amount.keys()):
        screen.blit(small_font.render(f"{player_bet_amount[p_id]} $", True, (255, 255, 255)), player_bet_positions[p_id])

def place_pot(screen, pot_amount):
    screen.blit(font.render(f"Pot: {pot_amount} $", True, (255, 255, 255)), pot_pos)

def place_cross(screen, folded_players, cross):
    for p_id in list(folded_players.keys()):
        if folded_players[p_id]:
            screen.blit(cross, card_positions[p_id])

def place_game_number(screen, game_n):
    screen.blit(font.render(f"Game: {game_n}", True, (255, 255, 255)), game_number_pos)

def place_paused(screen, paused):
    if paused:
        screen.blit(font.render(f"PAUSED", True, (255, 0, 0)), (0, 0))
 
bg = pygame.image.load("assets/Poker_table.png")
dealer_brick = pygame.image.load("assets/Dealer_brick.png")
cross = pygame.image.load("assets/cross.png")
table_folder = filedialog.askdirectory(initialdir=str(Path(os.getcwd()).parent.absolute()) + "/recorded_tables", title="Select a table-folder")
curr_game = 0
game_folder = os.path.join(table_folder, f"Game_{curr_game}")
player_bals = get_bals(game_folder, init=True)
player_cards, table_cards = get_cards(game_folder)
player_cards, table_cards = load_cards(player_cards, table_cards)
print(player_cards, table_cards)
dealer = get_dealer(game_folder)
actions = get_actions(game_folder)
state = "Pre-flop"
state_map = {0: "Pre-flop", 1: "Flop", 2: "Turn", 3: "River"}
rev_state_map = {v: k for k, v in state_map.items()}
pot_amount = 0
current_action = 0
newgame = False
paused = False

start_counter = 5

# Game loop.
while True:

    if newgame:
        newgame = False
        post_bals = get_bals(game_folder, init=False)
        for p_id in list(post_bals.keys()):
            player_bals[p_id] = post_bals[p_id]
        for p_id in list(folded_players.keys()):
            folded_players[p_id] = False
        curr_game += 1
        if os.path.exists(os.path.join(table_folder, f"Game_{curr_game}")):
            game_folder = os.path.join(table_folder, f"Game_{curr_game}")
        else:
            break

        check_player_bals = get_bals(game_folder, init=True)
        for p_id in list(player_bals.keys()):
            if player_bals[p_id] != 0.0 and player_bals[p_id] != check_player_bals[p_id]:
                print(f"PLAYER {p_id} BALANCE WAS {player_bals[p_id]} AFTER GAME {curr_game - 1} BUT WAS {check_player_bals[p_id]} BEFORE GAME {curr_game}????")
        
        player_cards, table_cards = get_cards(game_folder)
        player_cards, table_cards = load_cards(player_cards, table_cards)
        dealer = get_dealer(game_folder)
        actions = get_actions(game_folder)
        state = "Pre-flop"
        pot_amount = 0
        current_action = 0
        start_counter = 5
    
    screen.fill((0, 0, 0))
    screen.blit(bg, (0, 50))

    place_game_number(screen, curr_game)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONUP:
            paused = not paused

    
    place_player_cards(screen, player_cards)
    place_dealer_brick(screen, dealer, dealer_brick)

    #print(actions)
    #print(state)
    #print(current_action)
    if len(actions[state]) > current_action:
        action_to_perform = actions[state][current_action]
        #print(action_to_perform)

    
    # Do the action
    if start_counter > 0:
        start_counter -= 1
    else:
        if not paused:
            current_action += 1
            
            if action_to_perform.action_str == "Call" or action_to_perform.action_str == "Raise":
                player_bals[action_to_perform.player_id] = round(player_bals[action_to_perform.player_id] - action_to_perform.amount, 2)
                player_bet_amount[action_to_perform.player_id] = round(player_bet_amount[action_to_perform.player_id] + action_to_perform.amount, 2)

            if action_to_perform.action_str == "Fold":
                folded_players[action_to_perform.player_id] = True
        ac_str = get_action_str(action_to_perform)
        place_action_text(screen, action_to_perform.player_id, ac_str)

        place_cross(screen, folded_players, cross)

    place_paused(screen, paused)
        

    print(current_action)
    print(state)
    print(actions[state])
    print(action_to_perform)
    print()
    if current_action == len(actions[state]) and state != "River":
        for p_id in list(player_bet_amount.keys()):
            pot_amount = round(pot_amount + player_bet_amount[p_id], 2)
            #pot_amount += player_bet_amount[p_id]
            player_bet_amount[p_id] = 0
        state = state_map[rev_state_map[state] + 1]
        current_action = 0
        if len(actions[state]) == 0:
            newgame = True
    elif current_action == len(actions[state]) and state == "River":
        break

    place_players(screen, player_bals, curr_player=action_to_perform.player_id)
    place_bet_amount(screen, player_bet_amount)
    place_pot(screen, pot_amount)



    pygame.display.flip()
    fpsClock.tick(fps)