import numpy as np
import pandas as pd
from config.cofig import PROJECT_DIR


def get_jester_data():

    top_jokes = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/top_jokes.csv")
    reward_list = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/reward_list.csv")

    idx_item = np.load(f"{PROJECT_DIR}/dataset/jester/idx_item.npy")
    actions = [int(item) for item in idx_item]

    action_features = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/action_features.csv")

    user_features = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/user_features.csv")

    user_stream = np.load(f"{PROJECT_DIR}/dataset/jester/user_stream.npy")

    return top_jokes, reward_list, actions, action_features, user_features, user_stream
