import unittest, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game_logic.Company import Company
from  game_logic.Market import Market

class TestMarketFunctions(unittest.TestCase):
    def setUp(self):
        # Create a new Market instance before each test method
        self.market = Market()

    def test(self):
        company1 = Company("Company1", self.market)
        company2 = Company("Company2", self.market)
        company1.spawn_item("wood", 100)
        company1.place_order("limit", "sell","wood",50, 10)

        self.assertEqual(self.market.get_price_quote('wood',20,'buy'),200)
        
        company2.set_balance(200)
        company2.place_order("market", "buy", "wood", 20)
        self.assertEqual(company2.inventory["wood"], 20)
        self.assertEqual(company1.inventory["wood"], 80)
        self.assertEqual(self.market.buy_orders['wood'][0].amount, 30)
        self.assertEqual(company1.balance, 200)
        self.assertEqual(company2.balance, 0)


    def test_limit_orders(self):
        company1 = Company("Company1", self.market)
        company2 = Company("Company2", self.market)
        company1.spawn_item("wood", 100)
        company1.place_order("limit", "sell","wood",50, 10)
        company2.set_balance(500)
        company2.place_order("limit", "buy","wood",50, 10)
        #self.assertEqual(company1.inventory['wood'], 50)
        #self.assertEqual(company2.inventory['wood'], 50)
if __name__ == '__main__':
    unittest.main()
