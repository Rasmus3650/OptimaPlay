import random, uuid
import numpy as np
from opensimplex import OpenSimplex


class Resource():
    def __init__(self, type, start_amount, regen_rate=0, extraction_rate = 1):

        self.type = type
        self.amount = start_amount # The amount of resource left in a given tile
        self.regen_rate = regen_rate
        self.extraction_rate = extraction_rate

        self.renewable = type in ["Wood", "Food", "Water"]

    def regenerate(self):
        if self.renewable:
            self.amount += self.regen_rate

    def extract(self):
        if self.amount > 0:
            to_extract = min(self.extraction_rate, self.amount)
            self.amount -= to_extract
            #TODO Add this resource to the company's inventory
        else:
            to_extract = 0
        return to_extract

    def reprJSON(self):
        return dict(type=self.type, amount=self.amount, renewable=self.renewable, extractionRate=self.extraction_rate, regenRate=self.regen_rate)


class Building():
    def __init__(self, type, owner, x, y, level=1, max_storage=100) -> None:
        self.type = type
        self.owner = owner
        self.level = level
        self.max_storage = max_storage
        self.storage = 0
        self.production_rate = 10 * level
        self.x = x
        self.y = y

    def produce(self):
        self.storage += self.production_rate
    
    def reprJSON(self):
        return dict(type=self.type, owner=self.owner, level=self.level, storage=self.storage, max_storage=self.max_storage, production_rate=self.production_rate)

    def upgrade(self):
        self.level += 1
        self.production_rate = 10 * self.level




class LumberMill(Building):
    def __init__(self, owner, x, y, level=1):
        super().__init__("LumberMill", owner, level)
        self.production_rate = 10 * level

    # def produce(self):
    #     self.storage += self.production_rate
    #     #return {"Wood": self.production_rate}

    def reprJSON(self):
        parent_dict = super().reprJSON()
        #parent_dict.update({"production_rate": self.production_rate})
        return parent_dict


class Mine(Building):
    def __init__(self, owner, x, y,  level=1):
        super().__init__("Mine", owner, level)
        self.production_rate = 10 * level

    # def produce(self):
    #     self.storage += self.production_rate
    #     return {"Iron": self.production_rate}

    def reprJSON(self):
        parent_dict = super().reprJSON()
        #parent_dict.update({"production_rate": self.production_rate})
        return parent_dict
    

class Farm(Building):
    def __init__(self, owner, x, y,  level=1):
        super().__init__("Farm", owner, level)
        self.production_rate = 10 * level

    # def produce(self):
    #     return {"Food": self.production_rate}
    
    def reprJSON(self):
        parent_dict = super().reprJSON()
        #parent_dict.update({"production_rate": self.production_rate})
        return parent_dict

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
        self.biome_to_building_map = {"Forest": "LumberMill", "Mountain": "Mine"}
        self.building_to_biome_map = {"LumberMill": "Forest", "Mine": "Mountain"}
        
    
    
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
            build = Building("HQ", company, random_column, random_row)
            self.map[random_row][random_column].place_building(build)
            company.buildings.append(build)
            print(f"Spawning HQ at {random_row}, {random_column}")
    
    def spawn_building(self, building_type, owner):
        target_biome = self.building_to_biome_map[building_type]
        random_row = random.randint(0, self.height - 1)
        random_column = random.randint(0, self.width - 1)
        while self.map[random_row][random_column].biome != target_biome:
            random_row = random.randint(0, self.height - 1)
            random_column = random.randint(0, self.width - 1)
        to_build = Building(building_type, owner, random_column, random_row)
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

