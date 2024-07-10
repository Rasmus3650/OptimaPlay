from .Inventory import Inventory

class Building():
    def __init__(self, type, owner, x, y, tile, level=1, max_storage=100) -> None:
        self.type = type
        self.owner = owner
        self.level = level
        self.max_storage = round(max_storage * (level*0.5))
        if type == "HQ":
            self.inventory = owner.inventory
        else:
            self.inventory = Inventory()
        self.production_rate = round(10 * level)
        self.tile = tile
        self.x = x
        self.y = y
        self.type_to_resource_map = {"LumberMill": "Wood", "Mine": "Iron", "Farm": "Food"}
        self.resource_ratio = self.get_resources()

    def get_resources(self):
        resource_ratio = {}
        
        total_amount = 0
        for resource in list(self.tile.resources.keys()):
            total_amount += self.tile.resources[resource].amount
            resource_ratio[self.tile.resources[resource].type] = self.tile.resources[resource].amount
        
        for resource in list(resource_ratio.keys()):
            resource_ratio[resource] /= total_amount
        
        return resource_ratio

    def produce(self):
        item_name = self.type_to_resource_map[self.type]
        self.inventory.add_items(item_name, round(self.resource_ratio[item_name] * self.production_rate))
    
    def reprJSON(self):
        return dict(type=self.type, owner=self.owner, level=self.level, inventory=self.inventory, max_storage=self.max_storage, production_rate=self.production_rate)

    def upgrade(self):
        self.level += 1
        self.production_rate = 10 * self.level

class LumberMill(Building):
    def __init__(self, owner, x, y, tile, level=1):
        super().__init__("LumberMill", owner, x, y, tile, level)
        self.production_rate = 10 * level

    # def produce(self):
    #     self.storage += self.production_rate
    #     #return {"Wood": self.production_rate}

    def reprJSON(self):
        parent_dict = super().reprJSON()
        #parent_dict.update({"production_rate": self.production_rate})
        return parent_dict


class Mine(Building):
    def __init__(self, owner, x, y, tile, level=1):
        super().__init__("Mine", owner, x, y, tile, level)
        self.production_rate = 10 * level

    # def produce(self):
    #     self.storage += self.production_rate
    #     return {"Iron": self.production_rate}

    def reprJSON(self):
        parent_dict = super().reprJSON()
        #parent_dict.update({"production_rate": self.production_rate})
        return parent_dict
    

class Farm(Building):
    def __init__(self, owner, x, y, tile, level=1):
        super().__init__("Farm", owner, x, y, tile, level)
        self.production_rate = 10 * level

    # def produce(self):
    #     return {"Food": self.production_rate}
    
    def reprJSON(self):
        parent_dict = super().reprJSON()
        #parent_dict.update({"production_rate": self.production_rate})
        return parent_dict


class Forge(Building):
    def __init__(self, owner, x, y, tile, level=1):
        super().__init__("Forge", owner, x, y, tile, level)
        self.production_rate = 10 * level

    # def produce(self):
    #     self.storage += self.production_rate
    #     return {"Iron": self.production_rate}

    def reprJSON(self):
        parent_dict = super().reprJSON()
        #parent_dict.update({"production_rate": self.production_rate})
        return parent_dict