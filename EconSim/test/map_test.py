import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game_logic.terrain.Map import Map
from game_logic.Company import Company
from game_logic.Drone import Drone
from game_logic.Market import Market

# import unittest


# class TestMapFunctions(unittest.TestCase):
#     def setUp(self):
#         # Create a new Market instance before each test method
#         self.map = Map(50,50)

#     def terrain_test(self):
#         print(self.map)
        
# if __name__ == '__main__':
#     unittest.main()
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj,'reprJSON'):
            return obj.reprJSON()
        else:
            return json.JSONEncoder.default(self, obj)


def map_test(consumer_thread):
    map = Map(200,200)
    company1 = Company("1", None, 1)
    company2 = Company("2", None, 1)
    company3 = Company("3", None, 1)
    drone1 = Drone(owner=company1)
    map.spawn_headquarters([company1, company2, company3])
    lumbermill = map.spawn_building("LumberMill", company1)
    lumbermill.produce()
    lumbermill.produce()
    lumbermill.produce()
    lumbermill.produce()
    drone1.transport(lumbermill, company1.buildings[0])
    
    #print(map)
    save = {}
    save['map'] = json.dumps(map.reprJSON(), cls=ComplexEncoder)
    save['company1'] = json.dumps(company1.reprJSON(), cls=ComplexEncoder)
    save['company2'] = json.dumps(company2.reprJSON(), cls=ComplexEncoder)
    save['company3'] = json.dumps(company3.reprJSON(), cls=ComplexEncoder)
    logger = consumer_thread
    logger.enqueue_data(save, "EconSim/recorded_tables/table_1/Game_0")


