import random


def simulate_breakdown(p_net):
    inputs={"p_net":p_net}
    scenarios=[{"name":"Battery Failure","type":"Electrical","input":"p_net","duration":10*60,"prob": lambda s: 0 if s <= 2000 else (1.0 if s >= 4100 else 0.05 + 0.95 * ((s - 2000) / 2100) ** 3)}]
    seed=random.random()
    stop_time=0
    for scenario in scenarios:
        if seed < scenario["prob"](inputs[scenario["input"]]):
            stop_time+=scenario["duration"]
            break
    return stop_time

