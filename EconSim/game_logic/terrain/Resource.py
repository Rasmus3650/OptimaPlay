


class Resource():
    def __init__(self, type, start_amount, regen_rate=0):

        self.type = type
        self.amount = start_amount # The amount of resource left in a given tile
        self.regen_rate = regen_rate

        self.renewable = type in ["Wood", "Food", "Water"]

    def regenerate(self):
        if self.renewable:
            self.amount += self.regen_rate

    def reprJSON(self):
        return dict(type=self.type, amount=self.amount, renewable=self.renewable, regenRate=self.regen_rate)