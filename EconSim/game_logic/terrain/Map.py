import random, uuid, arcade
import numpy as np
from opensimplex import OpenSimplex


class Resource():
    def __init__(self, type, start_amount):

        self.type = type
        self.amount = start_amount # The amount of resource left in a given tile


        self.renewable = False
        if type == "Wood" or type == "Food":
            self.renewable = True

    def regenerate(self, type):
        # Regenerate a given resource over time (different for each resource)
        pass

    def reprJSON(self):
        return dict(type=self.type, amount=self.amount, renewable=self.renewable)

class Building():
    def __init__(self, type, owner) -> None:
        self.type = type
        self.owner = owner
    
    def reprJSON(self):
        return dict(type=self.type, owner=self.owner)


class Tile():
    def __init__(self, biome, resources):
        self.biome = biome
        self.resources = resources
        self.entities = {}      # All living entities (workers?) residing in a cell
        self.building = None    # Hold information about the building on a cell, if there is one

    def place_building(self, building):
        self.building = building
    
    def reprJSON(self):
        return dict(biome=self.biome, resources=self.resources, entities=self.entities, building=self.building)

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
            if random.random() >= 0.5:
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
            self.map[random_row][random_column].place_building(Building("HQ", company))
    
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

