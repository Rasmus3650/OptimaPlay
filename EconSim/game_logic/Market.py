class Order:
    def __init__(self, company, item, amount, price, order_type):
        self.company = company
        self.item = item
        self.amount = amount
        self.price = price
        self.order_type = order_type

class Transaction:
    def __init__(self, order_type, company, item, amount, price):
        self.order_type = order_type  # 'buy' or 'sell'
        self.company = company
        self.item = item
        self.amount = amount
        self.price = price

class TransactionLog:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction):
        self.transactions.append(transaction)

    def get_transactions(self):
        return self.transactions

class Market:
    def __init__(self):
        self.sell_orders = {}  # dict containing item as key and a list of sell orders
        self.buy_orders = {}   # dict containing item as key and a list of buy orders
        self.transaction_log = TransactionLog()


    

    def create_order(self, company, item, amount, price, direction):
        #TODO Need limit orders to trigger each other when relevant
        if direction == "buy":
            if item not in self.sell_orders:
                self.sell_orders[item] = []
            order = Order(company, item, amount, price, direction)
            self.sell_orders[item].append(order)
        if direction == "sell":
            if item not in self.buy_orders:
                self.buy_orders[item] = []
            order = Order(company, item, amount, price, direction)
            self.buy_orders[item].append(order)

    def get_price_quote(self, item, amount, direction):
        if direction == 'sell':
            orders = self.sell_orders
        elif direction == 'buy':
            orders = self.buy_orders
        else:
            raise ValueError(f"Invalid order type: {direction}")

        if item not in orders or not orders[item]:
            return 0

        filled_amount = 0
        total_price = 0
        order_list_copy = orders[item][:]
        for order in order_list_copy:
            if order.amount <= amount - filled_amount:
                filled_amount += order.amount
                total_price += order.amount * order.price
            else:
                partial_fill = amount - filled_amount
                filled_amount += partial_fill
                total_price += partial_fill * order.price
                break

        return total_price


    def fill_market_order(self, buyer, item, amount, direction):
        if direction == 'buy':
            orders = self.buy_orders
            inventory_adjustment = -1
        elif direction == 'sell':
            orders = self.sell_orders
            inventory_adjustment = 1
        else:
            raise ValueError(f"Invalid order type: {direction}")

        if item not in orders or not orders[item]:
            raise ValueError(f"No {direction} orders available for {item}")

        filled_amount = 0
        total_price = 0
        order_list_copy = orders[item][:]
        for order in order_list_copy:
            if order.amount <= amount - filled_amount:
                filled_amount += order.amount
                total_price += order.amount * order.price
                self.transaction_log.add_transaction(
                    Transaction(direction, order.company, item, order.amount, order.price)
                )
                if item in order.company.inventory:
                    order.company.inventory[item] += order.amount* inventory_adjustment
                else:
                    order.company.inventory[item] = order.amount*inventory_adjustment
                order.company.balance -= order.amount * order.price*inventory_adjustment
                
                if item in buyer.inventory:
                    buyer.inventory[item] += -1*(order.amount* inventory_adjustment)
                else:
                    buyer.inventory[item] = -1*(order.amount*inventory_adjustment)
                buyer.balance -= -1*(order.amount * order.price*inventory_adjustment)
                orders[item].remove(order)
            else:
                partial_fill = amount - filled_amount
                filled_amount += partial_fill
                total_price += partial_fill * order.price

                if item in order.company.inventory:
                    order.company.inventory[item] +=  partial_fill * inventory_adjustment
                else:
                    order.company.inventory[item] =  partial_fill * inventory_adjustment
                order.company.balance -= partial_fill * order.price * inventory_adjustment

                if item in buyer.inventory:
                    buyer.inventory[item] += -1*(partial_fill * inventory_adjustment)
                else:
                    buyer.inventory[item] = -1*(partial_fill * inventory_adjustment)
                buyer.balance -= -1*(partial_fill * order.price * inventory_adjustment)

                order.amount -= partial_fill
                self.transaction_log.add_transaction(
                    Transaction(direction, order.company, item, amount-filled_amount, order.price)
                )
                break

        if filled_amount > 0:
            price_per_unit = total_price / filled_amount
            return filled_amount, price_per_unit
        else:
            return 0, 0

