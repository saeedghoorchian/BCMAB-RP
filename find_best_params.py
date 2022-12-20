"""Script to find best BCMAB-RP params across dimensions"""
import json
import numpy as np
from pathlib import Path

# Enter the dimensions for the dataset you use now. These are for Jester.
DIMENSIONS = [15, 30, 60, 150, 300]
NUM_PARAMS = 72

all_rewards = np.zeros((len(DIMENSIONS), NUM_PARAMS))
for i, dim in enumerate(DIMENSIONS):
    p = Path('.')
    paths = list(p.glob(f'*bcmabrp*d_{dim}.json'))
    values = []
    for pat in paths:
        with open(pat, 'r') as f:
            tune_dict = json.load(f)
        for v in tune_dict:
            values.append(v)
    values = sorted(values, key=lambda x: x[0])
    keys = []
    for j, (k,v) in enumerate(values):
        keys.append(k)
        all_rewards[i, j] = v['Total reward mean']

mean_rewards = np.mean(all_rewards, axis=0)
best_ind = np.argmax(mean_rewards)
best_param = keys[best_ind]