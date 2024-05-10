from .Person import Person, Action
class Company():
    def __init__(self, name:str="", market=None, workers:int=1):
        self.inventory = {}
        self.balance = 0
        self.company_name = name
        self.market = market
        self.workers = {}
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

    def spawn_item(self, item, amount):
            if item in self.inventory:
                self.inventory[item] += amount
            else:
                self.inventory[item] = amount

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



