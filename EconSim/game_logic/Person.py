import names
import random
import hashlib

class Action():
    def __init__(self, action_type, target) -> None:
        self.action_type = action_type
        self.target = target

    def reprJSON(self):
        return dict(action_type = self.action_type, target = self.target)

class Person():
    def __init__(self):
        self.gender = "Female"
        if random.random() >= 0.5:
            self.gender = "Male"
        
        self.name = names.get_full_name(self.gender.lower())
        self.id = self.generate_id()
        self.skills = None  # Some RimWorld type of efficiency?
        self.action = Action("Idle", None)
        self.max_workload = 100 
        self.potential_work_left = self.max_workload

    def reprJSON(self):
        return dict(gender = self.gender, name = self.name, id = self.id, skills = self.skills, action = self.action, max_workload = self.max_workload, potential_work_left = self.potential_work_left)

    def generate_id(self):
        salt = str(random.randint(0, 1000000))
        data = f"{self.name}{self.gender}{salt}"
        hashed_id = hashlib.sha256(data.encode()).hexdigest()
        return hashed_id