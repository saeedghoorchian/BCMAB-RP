import os
import random
import numpy as np
import pandas as pd
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split

from config.cofig import PROJECT_DIR


THRESHOLD = None

random.seed(42)


def users_that_rated_all_items(ratings, items):
    user_sets = []
    for item_id in items:
        item_users = set(ratings[ratings.item_id == item_id].user_id)
        user_sets.append(item_users)
    return set.intersection(*user_sets)


def users_that_rated_all_top_k_items(ratings, k):
    items = ratings.groupby('item_id').size().sort_values(ascending=False)
    top_k_items = set(items[:k].index)
    top_k_ratings = ratings[ratings.item_id.isin(top_k_items)]
    top_k_users = users_that_rated_all_items(top_k_ratings, top_k_items)
    return top_k_users


def make_user_stream(user_features, times):
    """This function makes a sequence of random users.
    In the policy_evaluation function, users arrive according to this sequence
    """

    user_stream = []

    users_all = user_features["user_id"].unique()
    print(f"---\nThere are {len(users_all)} unique users in the user stream\n---")
    for t in range(times):
        rand_user_idx = random.randint(0, len(users_all) - 1)
        user_t = users_all[rand_user_idx]
        user_stream.append(user_t)
    return pd.DataFrame(data=user_stream, columns=["user_id"])


def main_data():
    os.makedirs(f"{PROJECT_DIR}/dataset/jester", exist_ok=True)

    jokes = Dataset.load_builtin(name='jester')

    # Shift all ratings to start at 0 (for NDCG computation)
    # min_rating = min([row[2] for row in jokes.raw_ratings])
    for i, row in enumerate(jokes.raw_ratings):
        rowl = list(row)
        # Min rating is -10
        rowl[2] += 10
        jokes.raw_ratings[i] = tuple(rowl)

    # svd_testset is basically zero, so that we can have all test_user_ids in the train set (fully labeled users)
    trainset, svd_testset = train_test_split(jokes, test_size=1e-10)

    # Only if you are running it the first time
    svd = SVD(n_factors=150, n_epochs=10, lr_all=0.005, reg_all=0.4)  # SVD(n_factors=100)
    svd.fit(trainset)

    # predictions = svd.test(svd_testset)
    # print(f"RMSE of SVD: {accuracy.rmse(predictions)}")
    print("Testing RMSE of SVD turned off, if you want to check it, switch it on and then back off again.")

    # SVD returns memory-views (cython).
    pu_all, qi_all = np.asarray(svd.pu), np.asarray(svd.qi)
    bu_all, bi_all = np.asarray(svd.bu), np.asarray(svd.bi)

    # Items
    idx_item = []
    for i in range(trainset.n_items):
        idx_item.append(trainset.to_raw_iid(i))
    idx_item_int = [int(item) for item in idx_item]
    print(f"#items in train set: {len(idx_item)}")

    np.save(f"{PROJECT_DIR}/dataset/jester/idx_item", idx_item)

    action_features = pd.DataFrame(data=qi_all)
    action_features.insert(0, 'item_id', idx_item_int)  # action_features["MovieID"] = idx
    action_features.to_csv(f"{PROJECT_DIR}/dataset/jester/action_features.csv", index=False)

    action_biases = pd.DataFrame(data=bi_all)
    action_biases.insert(loc=0, column='item_id', value=idx_item_int)
    action_biases.to_csv(f"{PROJECT_DIR}/dataset/jester/action_biases.csv", index=False)

    # Users
    idx_user = []
    for i in range(trainset.n_users):
        idx_user.append(trainset.to_raw_uid(i))
    idx_user_int = [int(item) for item in idx_user]
    print(f"#users in train set: {len(idx_user)}")

    ratings = pd.DataFrame(data=list(jokes.raw_ratings))
    ratings.columns = ["user_id", "item_id", "rating", "other"]
    del ratings["other"]

    test_user_ids = users_that_rated_all_top_k_items(ratings, 140)
    assert len(test_user_ids) == 5
    test_user_ids = list(map(int, test_user_ids))
    np.save(f"{PROJECT_DIR}/dataset/jester/test_user_ids", np.array(test_user_ids))

    user_features = pd.DataFrame(data=pu_all)
    user_features.insert(0, 'user_id', idx_user_int)

    user_stream = make_user_stream(user_features, 200000)
    user_stream.to_csv(f"{PROJECT_DIR}/dataset/jester/user_stream.csv", index=False)

    # Only save user features for those users that are present in the experiment.
    user_features = user_features[user_features.user_id.isin(set(user_stream["user_id"]))]
    user_features.to_csv(f"{PROJECT_DIR}/dataset/jester/user_features.csv", index=False)

    user_biases = pd.DataFrame(data=bu_all)
    user_biases.insert(loc=0, column='user_id', value=idx_user_int)
    # Only save user biases for those users that are present in the experiment.
    user_biases = user_biases[user_biases.user_id.isin(set(user_stream["user_id"]))]
    user_biases.to_csv(f"{PROJECT_DIR}/dataset/jester/user_biases.csv", index=False)

    # Ratings
    ratings = jokes.raw_ratings
    all_ratings = pd.DataFrame(data=ratings)
    all_ratings.columns = ['user_id', 'item_id', 'rating', 'other']
    del all_ratings['other']  # all_ratings.drop('other', axis=0)

    # Sort by the number of ratings for each joke
    actions_all = all_ratings.groupby('item_id').size().sort_values(ascending=False)  # [:140]

    actions_all = list(actions_all.index)
    # len(actions_all)
    top_jokes = all_ratings[all_ratings['item_id'].isin(actions_all)]

    print(f"User-rating matrix filled to total ratio: {len(top_jokes) / (len(idx_user) * len(idx_item))}")

    if THRESHOLD is not None:
        top_jokes['reward'] = np.where(top_jokes['rating'] >= THRESHOLD, 1, 0)
    else:
        # + 10 due to shifting done for the NDCG, see beginning of this function
        top_jokes['reward'] = np.where(top_jokes['rating'] >= 0 + 10, 1, 0)  # if original rating >=0, the user will like the joke

    top_jokes["item_id"] = pd.to_numeric(top_jokes["item_id"])
    top_jokes["user_id"] = pd.to_numeric(top_jokes["user_id"])
    top_jokes.to_csv(f"{PROJECT_DIR}/dataset/jester/top_jokes.csv")

    # Only save User-Joke pairs where the reward is 1.
    reward_list = top_jokes[['user_id', 'item_id', 'reward']]
    reward_list = reward_list[reward_list['reward'] == 1]
    if THRESHOLD is not None:
        reward_list.to_csv(f"{PROJECT_DIR}/dataset/jester/reward_list_{THRESHOLD}.csv", index=False)
    else:
        reward_list.to_csv(f"{PROJECT_DIR}/dataset/jester/reward_list.csv", index=False)

    # Save original ratings for NDCG calculation
    ratings_list = top_jokes[['user_id', 'item_id', 'rating']]
    ratings_list.to_csv(f"{PROJECT_DIR}/dataset/jester/ratings_list.csv", index=False)


if __name__ == '__main__':
    # for t in list(range(0, 21)):
    #     THRESHOLD = t
    main_data()
