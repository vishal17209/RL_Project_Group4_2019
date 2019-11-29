import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def qHeatmap(history_action_values):
    last_action_value = history_action_values[-1]
    action_states = list(last_action_value.keys())

    for action_value in history_action_values:
        for state in action_states:
            if(state not in action_value):
                action_value[state] = 0

    states = set([s for (s,a) in action_states])
    actions = set([a for (s,a) in action_states])

    encode_state = {}
    encode_actions = {}

    state_size = 0
    for s in states:
        encode_states[s] = state_size
        state_size+=1

    action_size = 0
    for a in actions:
        encode_actions[a] = action_size
        action_size+=1

    

    i = 1
    while(i < len(history_action_values)):

        i*=10