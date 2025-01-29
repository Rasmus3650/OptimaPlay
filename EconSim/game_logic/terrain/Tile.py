


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