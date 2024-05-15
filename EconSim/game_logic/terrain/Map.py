import random, uuid
import numpy as np
from opensimplex import OpenSimplex
from .Tile import Tile
from .Resource import Resource
import sys, os
from game_logic.Building import *


class Map():
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed
        if seed == None:
            self.seed = self.generate_seed()

        self.map = self.generate_world()
        self.all_biomes = ['Forest', 'Mountain', 'Plains', 'Beach', 'Ocean']
        self.all_resources = ['Wood', 'Water', 'Stone', 'Iron', 'Food']
        self.biome_to_building_map = {"Forest": "LumberMill", "Mountain": "Mine"}
        self.building_to_biome_map = {"LumberMill": "Forest", "Mine": "Mountain"}
        self.building_map = {"LumberMill": LumberMill, "Mine": Mine, "Farm": Farm, "HQ": Building}
    
    def generate_seed(self):
        return (uuid.uuid1().int >> 64)

    def generate_ressource(self, biome):
        # Make this probabilistic
        result = {}
        if biome == "Ocean":
            result['Water'] = Resource('Water', 10000)
        elif biome == "Forest":
            result['Wood'] = Resource('Wood', 1000)
            result['Food'] = Resource('Food', 200)
        elif biome == "Mountain":
            result['Stone'] = Resource('Stone', 1000)
            #if random.random() >= 0.5:
            result['Iron'] = Resource('Iron', 250)
        elif biome == "Plains":
            result['Food'] = Resource('Food', 500)
            result['Wood'] = Resource('Wood', 500)
        elif biome == "Beach":
            result['Food'] = Resource('Food', 750)
        return result


    def generate_world(self):
        world_map = []
        tmp = OpenSimplex(self.seed)
        random_offset= random.randint(1,500)
        for x in range(self.width):
            world_map.append([])
            for y in range(self.height):
                noise_value = tmp.noise2(x=(x+random_offset)*0.05, y=(y+random_offset)*0.05)
                biome = None
                if noise_value < 0:
                    biome = 'Ocean'
                elif noise_value >= 0 and noise_value < 0.15:
                    biome = 'Beach'
                elif noise_value >= 0.15 and noise_value < 0.4:
                    biome = 'Plains'
                elif noise_value >= 0.4 and noise_value < 0.65:
                    biome = "Forest"
                elif noise_value >= 0.65:
                    biome = 'Mountain'
                resources = self.generate_ressource(biome)
                world_map[x].append(Tile(biome, resources))
        return world_map
    
    def spawn_headquarters(self, companies):
        for company in companies:
            random_row = random.randint(0, self.height - 1)
            random_column = random.randint(0, self.width - 1)
            while self.map[random_row][random_column].biome == "Ocean":
                random_row = random.randint(0, self.height - 1)
                random_column = random.randint(0, self.width - 1)
            build = Building("HQ", company, random_column, random_row, self.map[random_row][random_column])
            self.map[random_row][random_column].place_building(build)
            company.buildings["HQ"] = [build]
            print(f"Spawning company {company.company_name}'s HQ at {random_row}, {random_column}")
    
    def spawn_building(self, building_type, owner):
        target_biome = self.building_to_biome_map[building_type]
        random_row = random.randint(0, self.height - 1)
        random_column = random.randint(0, self.width - 1)
        while self.map[random_row][random_column].biome != target_biome:
            random_row = random.randint(0, self.height - 1)
            random_column = random.randint(0, self.width - 1)
        to_build = self.building_map[building_type](owner, random_column, random_row, self.map[random_row][random_column])
        self.map[random_row][random_column].place_building(to_build)
        print(f"Spawning {building_type} at {random_row}, {random_column}")
        return to_build
    
    def __repr__(self):
        representation = ""
        biome_colors = {
            'Ocean': '\033[94m',  # Blue
            'Beach': '\033[93m',  # Yellow
            'Plains': '\033[92m',  # Green
            'Forest': '\033[22m',  # Dark Green
            'Mountain': '\033[90m'  # Grey
        }

        for row in self.map:
            for tile in row:
                if tile.building == None:
                    representation += biome_colors[tile.biome] + '#'  # Use '#' as a placeholder for tiles
                else:
                    representation += '\033[41m'+"#"#\e[0;31m
            representation += '\n'
        return representation + '\033[0m'

    def reprJSON(self):
        return dict(width=self.width, height=self.height, seed=self.seed, map=self.map, all_biomes=self.all_biomes, all_resources=self.all_resources)

