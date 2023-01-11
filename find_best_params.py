"""Script to find best BCMAB-RP params across dimensions"""
import json
import numpy as np
from pathlib import Path

# Enter the dimensions for the dataset you use now. These are for Jester.
DIMENSIONS = [25, 50, 100, 250, 500]
NUM_PARAMS = 72

all_rewards = np.zeros((len(DIMENSIONS), NUM_PARAMS))
for i, dim in enumerate(DIMENSIONS):
    p = Path('.')
    paths = list(p.glob(f'*cbrap*d_{dim}.json'))
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


# Additional for deepfm calculations
p = Path('.')
paths = list(p.glob('results_jester_config_*_t_30000_n_3_d_300.json'))
rewards = np.zeros(len(paths))
for pat in paths:
    ind = int(str(pat).split('config_')[1].split('_t_30000')[0])
    with open(pat, 'r') as f:
        tune_dict = json.load(f)
        print(ind, tune_dict[0][1]['Total reward mean'])
        rewards[ind] = tune_dict[0][1]['Total reward mean']
best_reward = rewards.max()
best_ind = rewards.argmax()