import json
from matplotlib import pyplot as plt

def plot_shot_results(job):
    y_pulse_results = job.result(timeout=120)
    measure_list = y_pulse_results.get_counts()
    zero_list = []
    sum_list = []
    for prob in measure_list:
        s = 0.
        for x in prob:
            s+=prob[x]
        sum_list.append(s)
        try:
            zero_list.append(prob['0'])
        except:
            zero_list.append(0)
    plt.plot(sum_list)
    plt.plot(zero_list, '.')
    plt.show()

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

def save_value(key : str, value):
    with open("saved_data.json") as f:
        json_data = json.load(f)
    json_data[key] = value
    with open("saved_data.json", 'w', encoding = 'UTF-8') as f:
        json.dump(json_data, f, indent = 8)
        
def load_value(key : str):
    with open("saved_data.json") as f:
        json_data = json.load(f)
    return json_data[key]

# schedule splitter 