import numpy as np
class OptimalControl:
    def __init__(self,):
        # ex) gate['X'] = X gate
        # gate.keys = ['X', 'X/2', 'Y', 'Y/2'] 
        # X/2 maens half pi pulse
        self.gate = {}
        self.ideal_gate = {}
        self.set_ideal_gate()
    
    def make_pulse_list(self, pulse_code_list):
        pulse_list = []
        for code in pulse_code_list:
            pulse_list.append(self.gate[code])
        return pulse_list
    
    def set_ideal_gate(self, ):
        self.ideal_gate['X'] = np.array([[0, 1], [1, 0]])