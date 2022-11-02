import numpy as np
import pandas as pd
from config.cofig import PROJECT_DIR
from data_loading import RecommenderDataset


def get_jester_data(threshold=None):

    top_jokes = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/top_jokes.csv")
    if threshold is None:
        reward_list = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/reward_list.csv")
    else:
        reward_list = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/reward_list_{threshold}.csv")
    ratings_list = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/ratings_list.csv")

    idx_item = np.load(f"{PROJECT_DIR}/dataset/jester/idx_item.npy")
    actions = [int(item) for item in idx_item]

    action_features = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/action_features.csv")
    action_biases = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/action_biases.csv")

    user_features = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/user_features.csv")
    user_biases = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/user_biases.csv")

    user_stream = pd.read_csv(f"{PROJECT_DIR}/dataset/jester/user_stream.csv")

    dataset = RecommenderDataset(
        actions=actions,
        action_features=action_features,
        action_biases=action_biases,
        user_stream=user_stream,
        true_user_features=user_features,
        user_features=user_features,
        user_biases=user_biases,
        reward_list=reward_list,
        ratings_list=ratings_list,
        ratings_range=(0.0, 20.0),
        implicit_feedback=False,
    )
    return dataset
