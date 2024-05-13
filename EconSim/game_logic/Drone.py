from .Inventory import Inventory



class Drone():
    def __init__(self, owner, level=1) -> None:
        self.owner = owner
        self.level = level
        self.inventory = Inventory()
        self.capacity = 10 * self.level
    
    def transport(self, start_building, end_building, item = None):
        if item is None:
            if start_building.resource_prod is not None:
                item = start_building.resource_prod
        start_amount = start_building.inventory.get_item_amount(item)
        if start_amount == 0:
            print(f"Drone from company {self.owner.company_name} couldn't transport {item} from building {start_building}")
            return False
        
        to_transport = min(self.capacity, start_amount)

        start_building.inventory.transfer_items(item, to_transport, self.inventory)
        # FLY
        self.inventory.transfer_items(item, to_transport, end_building.inventory)

        # cat = self.owner.crafter.get_category(item)
        # if cat in list(end_building.inventory.keys()):
        #     if item in list(end_building.inventory[cat]):
        #         end_building.inventory[cat][item] += to_transport
        #     else:
        #         end_building.inventory[cat][item] = to_transport
        # else:
        #     end_building.inventory[cat] = {item: to_transport}

        return True

        