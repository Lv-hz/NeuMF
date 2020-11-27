import numpy as np
from MLP.MLP import MLP

class Train():
    def __init__(self):
        self.model_init()
    
    def train(self, guid):
        self.model = MLP().mlp(guid)
