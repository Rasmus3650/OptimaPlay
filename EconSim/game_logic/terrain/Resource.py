


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
            #TODO Add this resource to the buildings's storage
        else:
            to_extract = 0
        return to_extract

    def reprJSON(self):
        return dict(type=self.type, amount=self.amount, renewable=self.renewable, extractionRate=self.extraction_rate, regenRate=self.regen_rate)