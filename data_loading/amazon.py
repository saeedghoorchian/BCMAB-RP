import numpy as np
import pandas as pd
from config.cofig import PROJECT_DIR


def get_amazon_data():
    actions = list(
        pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/actions.csv", sep="\t", header=0, engine="c")["item_id"]
    )

    action_features = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/action_features.csv")

    user_stream = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/user_stream.csv", sep="\t", header=0, engine="c")

    user_features = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/user_features.csv")

    reward_list = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/reward_list.csv", sep="\t", header=0, engine="c")

    return actions, action_features, user_stream, user_features, reward_list 
