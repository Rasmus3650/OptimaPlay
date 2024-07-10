from .Crafting import Crafting


class Inventory():
    def __init__(self) -> None:
        self.crafter = Crafting()
        self.inventory = {}
    
    def add_items(self, item_name, amount = 1):
        cat = self.crafter.get_category(item_name)
        if cat not in list(self.inventory.keys()):
            self.inventory[cat] = {item_name: amount}
        elif item_name not in list(self.inventory[cat].keys()):
            self.inventory[cat][item_name] = amount
        else:
            self.inventory[cat][item_name] += amount
    
    def remove_items(self, item_name, amount = 1, give_rest_if_not_enough = False):
        cat = self.crafter.get_category(item_name)
        if cat not in list(self.inventory.keys()) or item_name not in list(self.inventory[cat].keys()):
            return item_name, False
        if self.inventory[cat][item_name] < amount and not give_rest_if_not_enough:
            return item_name, False

        to_get = min(self.inventory[cat][item_name], amount)
        self.inventory[cat][item_name] -= to_get
        return item_name, to_get
    
    def transfer_items(self, item, amount, destination_inventory):
        i, a = self.remove_items(item, amount)
        destination_inventory.add_items(i, a)

    def get_item_names(self, category):
        if category not in list(self.inventory.keys()):
            return []
        return list(self.inventory[category].keys())
    
    def get_item_amount(self, item):
        cat = self.crafter.get_category(item)
        if cat not in list(self.inventory.keys()) or item not in list(self.inventory[cat].keys()):
            return 0
        return self.inventory[cat][item]
    
    def get_categories(self):
        return list(self.inventory.keys())

    def reprJSON(self):
        return self.inventory
    
    def __repr__(self) -> str:
        return str(self.inventory)