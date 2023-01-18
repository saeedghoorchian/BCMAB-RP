import argparse
import numpy as np
import pandas as pd
import random
import surprise as sp
from datetime import datetime
import time

from config.cofig import PROJECT_DIR, AMAZON_CONTEXT_DIMENSION


random.seed(42)


DEFAULT_AMAZON_RATINGS_PATH = f"{PROJECT_DIR}/dataset/amazon/Video_Games.csv"
AMAZON_NUMBER_OF_ACTIONS = 400

THRESHOLD = None

def print_mean_rewards(reward_list, actions):
    watched_list_series = (
        reward_list.groupby("user_id")["item_id"].agg(set=set).set
    )
    user_id_to_watched_list_index = {
        uid: ind for ind, uid in enumerate(watched_list_series.index)
    }
    reward_matrix = np.zeros((len(user_id_to_watched_list_index), len(actions)))
    for uid, ind in user_id_to_watched_list_index.items():
        watched_list = watched_list_series.iloc[ind]
        watched_indices = [i for i in range(len(actions)) if actions[i] in watched_list]
        # Binary vector of rewards for each arm.
        reward_matrix[ind, watched_indices] = 1

    mean_rew = reward_matrix.mean(axis=0)

    print(f"Reward of best 50 arms: {np.sort(mean_rew)[::-1][:50]}")
    print(f"Sum of rewards:\t {mean_rew.sum()}")

def create_actions_users_and_rewards(ratings_df):
    """Preprocess the original dataframe to extract actions, users and rewards from it.

    1. Extract top AMAZON_NUMBER_OF_ACTIONS items from the data (sort by number of ratings).
    2. Then extract users which rated those extracted items. This way, for every user at least one arm
    will have reward 1 later.
    3. Create rewards from ratings by considering rating > 3.0 being reward 1. Then only save user-item-reward triplets
    for rewards equal to 1. This way we can filter by user during evaluation and if item is there reward is 1.
    """
    # Choose a less sparse subset of data with a length of 1 year.
    YEARS_BEHIND = 5.8
    max_ts = ratings_df.timestamp.max()
    start_time = max_ts - 3600 * 24 * 365 * YEARS_BEHIND
    end_time = max_ts - 3600 * 24 * 365 * (YEARS_BEHIND-0.5)
    print(f"Start time: {datetime.fromtimestamp(start_time)}")
    print(f"End time: {datetime.fromtimestamp(end_time)}")
    ratings_df = ratings_df[ratings_df.timestamp > start_time]
    ratings_df = ratings_df[ratings_df.timestamp < end_time]

    actions = ratings_df.groupby("item_id").size().sort_values(ascending=False)[:AMAZON_NUMBER_OF_ACTIONS]
    actions = list(actions.index)

    # Only consider users that have watched some movies from the considered actions.
    top_ratings = ratings_df[ratings_df["item_id"].isin(actions)]
    top_ratings = top_ratings.sort_values("timestamp", ascending=1)
    user_stream = top_ratings[["user_id", "timestamp"]]

    NUM_UNIQUE_USERS = 7000
    top_users = list(top_ratings.groupby('user_id').size().sort_values(ascending=False)[:NUM_UNIQUE_USERS].index)
    user_stream = user_stream[user_stream.user_id.isin(top_users)]
    user_stream = user_stream.drop_duplicates(subset=['user_id'])
    users = set(user_stream.user_id)

    user_stream = user_stream.sample(n=135000, replace=True, random_state=42)

    print(f"Experiments has {len(actions)} items,\n{len(user_stream)} users\nof which {len(users)} are unique.")

    top_ratings = top_ratings[top_ratings.user_id.isin(users)]

    if THRESHOLD is not None:
        top_ratings["reward"] = np.where(top_ratings["rating"] >= THRESHOLD, 1, 0)
    else:
        top_ratings["reward"] = np.where(top_ratings["rating"] >= 0.0, 1, 0)
    reward_list = top_ratings[["user_id", "item_id", "reward", "rating"]]
    reward_list = reward_list[reward_list['reward'] == 1]
    print_mean_rewards(reward_list, actions)
    # Used for NDCG computation
    ratings_list = top_ratings[["user_id", "item_id", "reward"]]
    return top_ratings, actions, user_stream, reward_list, ratings_list



def preprocess_amazon_data(amazon_ratings_path):
    """Use surprise library to create user and item features and rewards from the original amazon ratings data."""
    ratings_df = pd.read_csv(amazon_ratings_path, names=["item_id", "user_id", "rating", "timestamp"])

    ratings_df, actions, user_stream, reward_list, ratings_list = create_actions_users_and_rewards(ratings_df)

    pd.DataFrame(actions, columns=["item_id"]).to_csv(f"{PROJECT_DIR}/dataset/amazon/actions.csv", sep='\t', index=False)
    if THRESHOLD is not None:
        reward_list.to_csv(f"{PROJECT_DIR}/dataset/amazon/reward_list_{THRESHOLD}.csv", sep='\t', index=False)
    else:
        reward_list.to_csv(f"{PROJECT_DIR}/dataset/amazon/reward_list.csv", sep='\t', index=False)
    ratings_list.to_csv(f"{PROJECT_DIR}/dataset/amazon/ratings_list.csv", sep='\t', index=False)

    # Use SVD to create user and item features.
    full_dataset = sp.Dataset.load_from_df(
        # order of columns matters here!
        ratings_df[["user_id", "item_id", "rating"]], reader=sp.Reader(rating_scale=(1.0, 5.0))
    )
    trainset, testset = sp.model_selection.train_test_split(full_dataset, test_size=1e-10)

    print(f"User-rating matrix filled to total ratio: {trainset.n_ratings / (trainset.n_items * trainset.n_users)}")

    # Dividing the context size by half because user and item context will be concatenated during evaluation.
    svd = sp.SVD(
        n_factors=AMAZON_CONTEXT_DIMENSION // 2, n_epochs=250, random_state=42,
        lr_all=0.01, reg_all=0.2,
    )
    svd.fit(trainset)

    # predictions = svd.test(testset)
    # print(f"RMSE of SVD: {sp.accuracy.rmse(predictions)}")

    # SVD returns memory-views (cython), hence asarray calls.
    pu_all, qi_all = np.asarray(svd.pu), np.asarray(svd.qi)
    bu_all, bi_all = np.asarray(svd.bu), np.asarray(svd.bi)

    trainset_item_ids = [trainset.to_raw_iid(i) for i in range(trainset.n_items)]
    items_that_have_features_set = set(trainset_item_ids)
    items_without_features_counter = sum([1 for item_id in actions if item_id not in items_that_have_features_set])
    print(f"Items without features: {items_without_features_counter}")

    action_features = pd.DataFrame(data=qi_all)
    action_features.insert(loc=0, column='item_id', value=trainset_item_ids)
    # Only save action features for actions present in the experiment
    action_features = action_features[action_features.item_id.isin(set(actions))]
    action_features.to_csv(f"{PROJECT_DIR}/dataset/amazon/action_features.csv", index=False)

    action_biases = pd.DataFrame(data=bi_all)
    action_biases.insert(loc=0, column='item_id', value=trainset_item_ids)
    action_biases = action_biases[action_biases.item_id.isin(set(actions))]
    action_biases.to_csv(f"{PROJECT_DIR}/dataset/amazon/action_biases.csv", index=False)


    trainset_user_ids = [trainset.to_raw_uid(i) for i in range(trainset.n_users)]
    users_that_have_features_set = set(trainset_user_ids)
    users_without_features_counter = sum(
        [1 for user_id in user_stream.user_id.unique() if user_id not in users_that_have_features_set]
    )
    print(f"Users without features: {users_without_features_counter}")
    # Only save users with features for the experiment.
    user_stream = user_stream[user_stream.user_id.isin(users_that_have_features_set)]

    user_stream.to_csv(f"{PROJECT_DIR}/dataset/amazon/user_stream.csv", sep='\t', index=False)

    user_features = pd.DataFrame(data=pu_all)
    user_features.insert(loc=0, column='user_id', value=trainset_user_ids)
    # Only save user features for those users that are present in the experiment.
    user_features = user_features[user_features.user_id.isin(set(user_stream.user_id))]
    user_features.to_csv(f"{PROJECT_DIR}/dataset/amazon/user_features.csv", index=False)

    user_biases = pd.DataFrame(data=bu_all)
    user_biases.insert(loc=0, column='user_id', value=trainset_user_ids)
    # Only save user biases for those users that are present in the experiment.
    user_biases = user_biases[user_biases.user_id.isin(set(user_stream.user_id))]
    user_biases.to_csv(f"{PROJECT_DIR}/dataset/amazon/user_biases.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        help="Path to a csv file containing Amazon user ratings.\n"
             "Must contain columns 'item_id', 'user_id', 'rating', 'timestamp' in that order.",
        required=True,
    )
    args = parser.parse_args()

    preprocess_amazon_data(args.data_path)
