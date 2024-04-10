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
    

class Visualizer():
    def __init__(self, fps=1) -> None:
        pygame.init()
        self.font = pygame.font.Font('freesansbold.ttf', 32)
        self.underline_font = pygame.font.Font('freesansbold.ttf', 32)
        self.underline_font.set_underline(True)
        self.small_font = pygame.font.Font('freesansbold.ttf', 16)
        self.player_positions = {0: (770, 55), 1: (1380, 110), 2: (1460, 685), 3: (770, 760), 4: (90, 685), 5: (160, 110)}
        self.card_positions = {0: (720, 190), 1: (1180, 240), 2: (1180, 560), 3: (720, 610), 4: (260, 560), 5: (260, 240)}
        self.card_x_offset = 80
        self.dealer_brick_positions = {0: (670, 190), 1: (1200, 200), 2: (1350, 560), 3: (885, 615), 4: (215, 570), 5: (215, 270)}
        self.action_text_pos = {0: (745, 20), 1: (1355, 80), 2: (1440, 655), 3: (740, 880), 4: (60, 655), 5: (125, 80)}
        self.player_bet_amount = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.player_bet_positions = {0: (790, 315), 1: (1165, 355), 2: (1165, 540), 3: (790, 590), 4: (410, 540), 5: (410, 355)}
        table_cards_y = 400
        self.table_cards_positions = {0: (602, table_cards_y), 1: (682, table_cards_y), 2: (762, table_cards_y), 3: (842, table_cards_y), 4: (922, table_cards_y)}
        self.pot_pos = (740, 365)
        self.game_number_pos = (300, 50)
        self.folded_players = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}
        #fps = 60
        self.fps = fps
        self.fpsClock = pygame.time.Clock()

        self.width, self.height = 1600, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.load_assets()
        


    def get_bals(self, game_folder_path, init=True):
        if init:
            bals_csv = open(os.path.join(game_folder_path, "InitBals.csv"))
        else:
            bals_csv = open(os.path.join(game_folder_path, "PostgameBals.csv"))
        players = bals_csv.readline().replace("\n", "").replace(" ", "").replace("P", "").split(",")
        bals = bals_csv.readline().replace("\n", "").replace(" ", "").split(",")
        player_dict = {int(k):float(v) for k, v in zip(players, bals)}
        return player_dict


    def place_players(self, player_bal_dict, curr_player = None):
        white = (255, 255, 255)
        green = (0, 255, 0)
        for player_id in list(player_bal_dict.keys()):
            if player_id not in list(self.player_positions.keys()): break
            if curr_player is not None and player_id == curr_player:
                self.screen.blit(self.underline_font.render(f"P {player_id}", True, green), self.player_positions[player_id])
            else:
                self.screen.blit(self.underline_font.render(f"P {player_id}", True, white), self.player_positions[player_id])
            self.screen.blit(self.small_font.render(str(player_bal_dict[player_id]) + " $", True, white), (self.player_positions[player_id][0], self.player_positions[player_id][1] + 40))

    def get_cards(self, game_folder_path):
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

    def load_cards(self, player_cards, table_cards, assets_folder="assets/Cards"):
        for p_id in list(player_cards.keys()):
            player_cards[p_id][0].append(pygame.image.load(os.path.join(assets_folder, player_cards[p_id][0][2])))
            player_cards[p_id][1].append(pygame.image.load(os.path.join(assets_folder, player_cards[p_id][1][2])))
        
        for c in table_cards:
            c.append(pygame.image.load(os.path.join(assets_folder, c[2])))

        return player_cards, table_cards

    def place_player_cards(self, player_cards):
        for p_id in list(player_cards.keys()):
            if p_id not in list(self.card_positions.keys()): break
            pos = self.card_positions[p_id]
            self.screen.blit(player_cards[p_id][0][3], pos)
            self.screen.blit(player_cards[p_id][1][3], (pos[0] + self.card_x_offset, pos[1]))

    def place_dealer_brick(self, dealer, dealer_brick):
        if dealer not in list(self.dealer_brick_positions.keys()): return
        self.screen.blit(dealer_brick, self.dealer_brick_positions[dealer])

            
    def get_dealer(self, game_folder_path):
        metadata_file = open(os.path.join(game_folder_path, "metadata.txt"))
        lines = metadata_file.readlines()
        metadata_file.close()
        for line in lines:
            if line[0:7] == "Dealer:":
                return int(line[8])

    def get_action_str(self, action):
        ac_str = f"{action.action_str}"
        if action.action_str == "Call" or action.action_str == "Raise":
            ac_str += f" {action.amount} $"
        return ac_str

    def get_actions(self, game_folder_path):
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

    def get_winner(self, game_folder_path):
        winner_file = open(os.path.join(game_folder_path, "Winners.csv"))
        lines = winner_file.readlines()
        winner_file.close()
        return int(lines[0][9])

    def place_action_text(self, player_id, text):
        self.screen.blit(self.small_font.render(f"{text}", True, (255, 255, 255)), self.action_text_pos[player_id])

    def place_bet_amount(self, player_bet_amount):
        for p_id in list(player_bet_amount.keys()):
            self.screen.blit(self.small_font.render(f"{player_bet_amount[p_id]} $", True, (255, 255, 255)), self.player_bet_positions[p_id])

    def place_pot(self, pot_amount):
        self.screen.blit(self.font.render(f"Pot: {pot_amount} $", True, (255, 255, 255)), self.pot_pos)

    def place_cross(self, folded_players, cross):
        for p_id in list(folded_players.keys()):
            if folded_players[p_id]:
                self.screen.blit(cross, self.card_positions[p_id])

    def place_game_number(self, game_n):
        self.screen.blit(self.font.render(f"Game: {game_n}", True, (255, 255, 255)), self.game_number_pos)

    def place_paused(self, paused):
        if paused:
            self.screen.blit(self.font.render(f"PAUSED", True, (255, 0, 0)), (0, 0))

    def place_winner(self, winner_id, winner_imgs, img_id):
        #print(winner_id, winner_imgs, img_id)
        self.screen.blit(winner_imgs[img_id], (self.player_positions[winner_id][0] - 45, self.player_positions[winner_id][1] - 45))
    
    def place_table_card(self, cards_on_table, table_cards_imgs):
        #print(cards_on_table)
        for i in range(cards_on_table):
            self.screen.blit(table_cards_imgs[i][3], self.table_cards_positions[i])
        
 
    def load_assets(self):
        self.bg = pygame.image.load("assets/Poker_table.png")
        self.dealer_brick = pygame.image.load("assets/Dealer_brick.png")
        self.cross = pygame.image.load("assets/cross.png")
        self.winner_circles = [pygame.image.load("assets/White_circle.png"), pygame.image.load("assets/Green_circle.png")]
    
    def replay_game(self, game_folder, curr_game):
        player_bals = self.get_bals(game_folder, init=True)
        player_cards, table_cards = self.get_cards(game_folder)
        player_cards, table_cards = self.load_cards(player_cards, table_cards)
        #print(player_cards, table_cards)
        dealer = self.get_dealer(game_folder)
        actions = self.get_actions(game_folder)
        state = "Pre-flop"
        state_map = {0: "Pre-flop", 1: "Flop", 2: "Turn", 3: "River"}
        rev_state_map = {v: k for k, v in state_map.items()}
        pot_amount = 0
        current_action = 0
        newgame = False
        paused = False

        start_counter = 5
        newgame_counter = -1
        newgame_counter_default = 5
        curr_winner = None
        w_img_id = 0

        dealing = False
        cards_on_table = 0
        cards_amount = {"Flop": 3, "Turn": 4, "River": 5}

        while True:
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.bg, (0, 50))

            if newgame_counter >= 0:
                
                if curr_winner is None:
                    curr_winner = self.get_winner(game_folder)
                #print(f"Winner: Player {curr_winner}")
                if newgame_counter == 0:
                    newgame = True
                    curr_winner = None
                else:
                    self.place_winner(curr_winner, self.winner_circles, w_img_id)
                newgame_counter -= 1
                
                w_img_id = (w_img_id + 1) % 2


            if newgame:
                newgame = False
                post_bals = self.get_bals(game_folder, init=False)
                for p_id in list(post_bals.keys()):
                    player_bals[p_id] = post_bals[p_id]
                for p_id in list(self.folded_players.keys()):
                    self.folded_players[p_id] = False
                break
                #curr_game += 1
                #if os.path.exists(os.path.join(table_folder, f"Game_{curr_game}")):
                #    game_folder = os.path.join(table_folder, f"Game_{curr_game}")
                #else:
                #    break

                check_player_bals = self.get_bals(game_folder, init=True)
                for p_id in list(player_bals.keys()):
                    if player_bals[p_id] != 0.0 and player_bals[p_id] != check_player_bals[p_id]:
                        print(f"PLAYER {p_id} BALANCE WAS {player_bals[p_id]} AFTER GAME {curr_game - 1} BUT WAS {check_player_bals[p_id]} BEFORE GAME {curr_game}????")
                
                player_cards, table_cards = self.get_cards(game_folder)
                player_cards, table_cards = self.load_cards(player_cards, table_cards)
                dealer = self.get_dealer(game_folder)
                actions = self.get_actions(game_folder)
                state = "Pre-flop"
                pot_amount = 0
                current_action = 0
                start_counter = 5
            
            if dealing and newgame_counter < 0:
                if cards_on_table < cards_amount[state]:
                    cards_on_table += 1
                else:
                    dealing = False
            
            #print(cards_on_table)
            self.place_table_card(cards_on_table, table_cards)

            self.place_game_number(curr_game)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONUP:
                    paused = not paused

            
            self.place_player_cards(player_cards)
            self.place_dealer_brick(dealer, self.dealer_brick)

            #print(actions)
            #print(state)
            #print(current_action)
            if len(actions[state]) > current_action and newgame_counter < 0:
                action_to_perform = actions[state][current_action]
                #print(action_to_perform)

            
            # Do the action
            if start_counter > 0:
                start_counter -= 1
            elif newgame_counter < 0 and not dealing:
                if not paused:
                    current_action += 1
                    
                    if action_to_perform.action_str == "Call" or action_to_perform.action_str == "Raise":
                        player_bals[action_to_perform.player_id] = round(player_bals[action_to_perform.player_id] - action_to_perform.amount, 2)
                        self.player_bet_amount[action_to_perform.player_id] = round(self.player_bet_amount[action_to_perform.player_id] + action_to_perform.amount, 2)

                    if action_to_perform.action_str == "Fold":
                        self.folded_players[action_to_perform.player_id] = True
                ac_str = self.get_action_str(action_to_perform)
                self.place_action_text(action_to_perform.player_id, ac_str)

                self.place_cross(self.folded_players, self.cross)

            self.place_paused(paused)
            
            if newgame_counter < 0 and not dealing:
                if current_action == len(actions[state]) and state != "River":
                    dealing = True
                    for p_id in list(self.player_bet_amount.keys()):
                        pot_amount = round(pot_amount + self.player_bet_amount[p_id], 2)
                        self.player_bet_amount[p_id] = 0
                    state = state_map[rev_state_map[state] + 1]
                    current_action = 0
                    if len(actions[state]) == 0:
                        newgame_counter = newgame_counter_default
                elif current_action == len(actions[state]) and state == "River":
                    newgame_counter = newgame_counter_default

            self.place_players(player_bals, curr_player=action_to_perform.player_id)
            self.place_bet_amount(self.player_bet_amount)
            self.place_pot(pot_amount)



            pygame.display.flip()
            self.fpsClock.tick(self.fps)

    def replay_table(self, table_folder):
        games = sorted(list(os.walk(table_folder))[0][1], key=lambda x: int(x.split('_')[1]))
        for game in games:
            
            game_folder = os.path.join(table_folder, game)
            self.replay_game(game_folder, int(game.split("_")[1]))

        


obj = Visualizer(fps=1)
#obj.replay_table("C:/Users/rune2/Documents/OptimaPlay_All/OptimaPlay/recorded_tables/table_1")
obj.replay_game("C:/Users/rune2/Documents/OptimaPlay_All/OptimaPlay/recorded_tables/table_1/Game_18", 18)
#table_folder = filedialog.askdirectory(initialdir=str(Path(os.getcwd()).parent.absolute()) + "/recorded_tables", title="Select a table-folder")
#curr_game = 0
#game_folder = os.path.join(table_folder, f"Game_{curr_game}")



# Game loop.
