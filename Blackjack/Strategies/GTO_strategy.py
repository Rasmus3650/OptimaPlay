import os, sys, random, math
from .strategy import Strategy

class GTO_strategy(Strategy):
    def __init__(self) -> None:
        self.hard_totals = {2: {8: "Hit", 9: "Hit", 10: "Double", 11: "Double", 12: "Hit", 13: "Stand", 14: "Stand", 15: "Stand", 16: "Stand", 17: "Stand"},
                   3: {8: "Hit", 9: "Double", 10: "Double", 11: "Double", 12: "Hit", 13: "Stand", 14: "Stand", 15: "Stand", 16: "Stand", 17: "Stand"},
                   4: {8: "Hit", 9: "Double", 10: "Double", 11: "Double", 12: "Stand", 13: "Stand", 14: "Stand", 15: "Stand", 16: "Stand", 17: "Stand"},
                   5: {8: "Hit", 9: "Double", 10: "Double", 11: "Double", 12: "Stand", 13: "Stand", 14: "Stand", 15: "Stand", 16: "Stand", 17: "Stand"},
                   6: {8: "Hit", 9: "Double", 10: "Double", 11: "Double", 12: "Stand", 13: "Stand", 14: "Stand", 15: "Stand", 16: "Stand", 17: "Stand"},
                   7: {8: "Hit", 9: "Hit", 10: "Double", 11: "Double", 12: "Hit", 13: "Hit", 14: "Hit", 15: "Hit", 16: "Hit", 17: "Stand"},
                   8: {8: "Hit", 9: "Hit", 10: "Double", 11: "Double", 12: "Hit", 13: "Hit", 14: "Hit", 15: "Hit", 16: "Hit", 17: "Stand"},
                   9: {8: "Hit", 9: "Hit", 10: "Double", 11: "Double", 12: "Hit", 13: "Hit", 14: "Hit", 15: "Hit", 16: "Hit", 17: "Stand"},
                   10: {8: "Hit", 9: "Hit", 10: "Hit", 11: "Double", 12: "Hit", 13: "Hit", 14: "Hit", 15: "Hit", 16: "Hit", 17: "Stand"},
                   11: {8: "Hit", 9: "Hit", 10: "Hit", 11: "Double", 12: "Hit", 13: "Hit", 14: "Hit", 15: "Hit", 16: "Hit", 17: "Stand"}
                   }
        self.pair_splitting = {2: {2: "No", 3: "No", 4: "No", 5: "No", 6: "No", 7: "Yes", 8: "Yes", 9: "Yes", 10: "No", 11: "Yes"},
                            3: {2: "No", 3: "No", 4: "No", 5: "No", 6: "Yes", 7: "Yes", 8: "Yes", 9: "Yes", 10: "No", 11: "Yes"},
                            4: {2: "Yes", 3: "Yes", 4: "No", 5: "No", 6: "Yes", 7: "Yes", 8: "Yes", 9: "Yes", 10: "No", 11: "Yes"},
                            5: {2: "Yes", 3: "Yes", 4: "No", 5: "No", 6: "Yes", 7: "Yes", 8: "Yes", 9: "Yes", 10: "No", 11: "Yes"},
                            6: {2: "Yes", 3: "Yes", 4: "No", 5: "No", 6: "Yes", 7: "Yes", 8: "Yes", 9: "Yes", 10: "No", 11: "Yes"},
                            7: {2: "Yes", 3: "Yes", 4: "No", 5: "No", 6: "No", 7: "Yes", 8: "Yes", 9: "No", 10: "No", 11: "Yes"},
                            8: {2: "No", 3: "No", 4: "No", 5: "No", 6: "No", 7: "No", 8: "Yes", 9: "Yes", 10: "No", 11: "Yes"},
                            9: {2: "No", 3: "No", 4: "No", 5: "No", 6: "No", 7: "No", 8: "Yes", 9: "Yes", 10: "No", 11: "Yes"},
                            10: {2: "No", 3: "No", 4: "No", 5: "No", 6: "No", 7: "No", 8: "Yes", 9: "No", 10: "No", 11: "Yes"},
                            11: {2: "No", 3: "No", 4: "No", 5: "No", 6: "No", 7: "No", 8: "Yes", 9: "No", 10: "No", 11: "Yes"}
                            }
        self.soft_totals = {2: {2: "Hit", 3: "Hit", 4: "Hit", 5: "Hit", 6: "Hit", 7: "Double", 8: "Stand", 9: "Stand"},
                        3: {2: "Hit", 3: "Hit", 4: "Hit", 5: "Hit", 6: "Double", 7: "Double", 8: "Stand", 9: "Stand"},
                        4: {2: "Hit", 3: "Hit", 4: "Double", 5: "Double", 6: "Double", 7: "Double", 8: "Stand", 9: "Stand"},
                        5: {2: "Double", 3: "Double", 4: "Double", 5: "Double", 6: "Double", 7: "Double", 8: "Stand", 9: "Stand"},
                        6: {2: "Double", 3: "Double", 4: "Double", 5: "Double", 6: "Double", 7: "Double", 8: "Double", 9: "Stand"},
                        7: {2: "Hit", 3: "Hit", 4: "Hit", 5: "Hit", 6: "Hit", 7: "Stand", 8: "Stand", 9: "Stand"},
                        8: {2: "Hit", 3: "Hit", 4: "Hit", 5: "Hit", 6: "Hit", 7: "Stand", 8: "Stand", 9: "Stand"},
                        9: {2: "Hit", 3: "Hit", 4: "Hit", 5: "Hit", 6: "Hit", 7: "Hit", 8: "Stand", 9: "Stand"},
                        10: {2: "Hit", 3: "Hit", 4: "Hit", 5: "Hit", 6: "Hit", 7: "Hit", 8: "Stand", 9: "Stand"},
                        11: {2: "Hit", 3: "Hit", 4: "Hit", 5: "Hit", 6: "Hit", 7: "Hit", 8: "Stand", 9: "Stand"}
                        }