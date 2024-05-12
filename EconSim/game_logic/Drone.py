




class Drone():
    def __init__(self, owner, level=1) -> None:
        self.owner = owner
        self.level = level
        self.capacity = 10 * self.level
    
    def transport(self, start_building, end_building):
        start_building.storage -= self.capacity
        end_building.storage += self.capacity

        