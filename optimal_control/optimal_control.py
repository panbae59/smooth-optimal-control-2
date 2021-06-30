class OptimalControl:
    def __init__(self,):
        # ex) gate['X'] = X gate
        # gate.keys = ['X', 'X/2', 'Y', 'Y/2'] 
        # X/2 maens half pi pulse
        self.gate = {}
    
    def make_pulse_list(self, pulse_code_list):
        pulse_list = []
        for code in pulse_code_list:
            pulse_list.append(self.gate[code])
        return pulse_list