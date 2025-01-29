


class Crafting():
    def __init__(self) -> None:
        self.buildings = {"LumberMill": {"Wood": 20, "Iron": 2},
                          "Mine": {},
                          "Farm": {},
                          "Forge": {}}
        self.drones = {"drone": {}}


        self.recepies = {"Building": self.buildings, "Drone": self.drones}


        self.categories = {"Basic": ["Wood", "Iron", "Water", "Food"], "Buildings": list(self.buildings.keys()), "Drones": list(self.drones.keys())}

    def get_category(self, item_name):
        for key in list(self.categories.keys()):
            if item_name in self.categories[key]:
                return key
        return None
