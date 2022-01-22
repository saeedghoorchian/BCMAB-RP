import numpy as np
import pandas as pd


def get_jester_data():

    top_jokes = pd.read_csv('dataset/jester/top_jokes.csv')
    reward_list = pd.read_csv('dataset/jester/reward_list.csv')

    idx_item = np.load('dataset/jester/idx_item.npy')
    actions = [int(item) for item in idx_item]

    action_features = pd.read_csv('dataset/jester/action_features.csv')

    user_features = pd.read_csv('dataset/jester/user_features.csv')

    user_stream = np.load('dataset/jester/user_stream.npy')

    return top_jokes, reward_list, actions, action_features, user_features, user_stream
