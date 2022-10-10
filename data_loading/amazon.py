import numpy as np
import pandas as pd
from config.cofig import PROJECT_DIR


def get_amazon_data(threshold=None):
    actions = list(
        pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/actions.csv", sep="\t", header=0, engine="c")["item_id"]
    )

    action_features = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/action_features.csv")
    action_biases = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/action_biases.csv")

    user_stream = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/user_stream.csv", sep="\t", header=0, engine="c")

    user_features = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/user_features.csv")
    user_biases = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/user_biases.csv")

    if threshold is None:
        reward_list = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/reward_list.csv", sep="\t", header=0, engine="c")
    else:
        reward_list = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/reward_list_{threshold}.csv", sep="\t", header=0, engine="c")

    ratings_list = pd.read_csv(f"{PROJECT_DIR}/dataset/amazon/ratings_list.csv", sep="\t", header=0, engine="c")

    return actions, action_features, action_biases, user_stream, user_features, user_biases, reward_list, ratings_list
