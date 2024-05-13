from .Person import Person, Action
from .Crafting import Crafting
from .Inventory import Inventory

class Company():
    def __init__(self, name:str="", market=None, workers:int=1, map = None):
        self.inventory = Inventory()
        self.balance = 0
        self.company_name = name
        self.market = market
        self.workers = {}
        self.buildings = {}
        self.map = map
        self.crafter = Crafting()
        self.hire_workers(workers)

    def reprJSON(self):
        return dict(inventory =  self.inventory, balance = self.balance, name = self.company_name, market = self.market, workers = self.workers)
    
    def hire_workers(self, amount):
        for _ in range(amount):
            worker = Person()
            self.workers[worker.name] = worker

    def get_idle_workers(self):
        result = []
        for worker in self.workers.items():
            if worker.action.action_type == "Idle":
                result.append(worker)
        return result

    def gather_resource(self, target, workers):
        idle_workers = self.get_idle_workers()

        for i in range(workers):
            idle_workers[i].action = Action("Gather", target)

    def gather_resource_amount(self, target, amount, workers = None):
        # Figure out how much work something takes


        # idle_workers = self.get_idle_workers()
        # if workers is None:
        #     for worker in idle_workers:

        pass

    def make_item():
        pass

    def build():
        pass

    def place_building(self, building_name): #Tænker der skal x, y som parametre her på et tidspunkt
        self.map.spawn_building(building_name, self)

    def craft(self, category, item_name):
        self.inventory = self.buildings["HQ"][0].inventory
        can_craft = True
        recepie = self.crafter.recepies[category][item_name]
        print(f"Recepie: {recepie}")
        for item in list(recepie.keys()):
            curr_cat = self.crafter.get_category(item)
            print(f"{curr_cat}: {item}")
            if curr_cat not in self.inventory.get_categories() or item not in self.inventory.get_item_names(curr_cat) or self.inventory.get_item_amount(item) < recepie[item]:
                can_craft = False
        
        if not can_craft:
            print(f"Company {self.company_name} tried to craft {item_name}, but has insufficient resources!")
            return False
        
        for item in list(recepie.keys()):
            curr_cat = self.crafter.get_category(item)
            self.inventory.remove_items(item, recepie[item])
            #self.inventory[curr_cat][item] -= recepie[item]

        self.inventory.add_items(item_name, 1)

        # if category not in list(self.inventory.keys()):
        #     self.inventory[category] = {item_name: 1}
        # elif item_name not in list(self.inventory[category].keys()):
        #     self.inventory[category][item_name] = 1
        # else:
        #     self.inventory[category][item_name] += 1
        return True


    def spawn_item(self, item, amount):
        self.inventory.add_items(item, amount)
        # if item in self.inventory:
        #     self.inventory[item] += amount
        # else:
        #     self.inventory[item] = amount

    def set_balance(self, amount):
        self.balance = amount

    def place_order(self, order_type:str, direction:str, item, amount, price=None):
        # order_type is limit/market
        # direction is buy/sell

        if order_type == "market":
            if self.market.get_price_quote(item, amount, direction) <= self.balance:
                self.market.fill_market_order(self, item, amount, direction)
        elif order_type == "limit":
            self.market.create_order(self, item, amount, price, direction)
        else:
             raise ValueError(f"{order_type} is not a valid order type, try limit or market")



