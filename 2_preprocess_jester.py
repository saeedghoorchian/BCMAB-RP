import random
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from config.cofig import PROJECT_DIR


def make_user_stream(user_features, times):
    """This function makes a sequence of random users.
    In the policy_evaluation function, users arrive according to this sequence
    """

    user_stream = []

    users_all = user_features["userid"].unique()
    for t in range(times):
        rand_user_idx = random.randint(0 ,len(users_all ) -1)
        user_t = users_all[rand_user_idx]
        user_stream.append(user_t)
    return user_stream


def main_data():
    jokes = Dataset.load_builtin(name='jester')
    trainset, testset = train_test_split(jokes, test_size=0.2)

    # Only if you are running it the first time
    svd = SVD(n_factors=150, n_epochs=10, lr_all=0.005, reg_all=0.4)  # SVD(n_factors=100)
    svd.fit(trainset)

    predictions = svd.test(testset)
    print(accuracy.rmse(predictions))

    pu_all, qi_all = svd.pu, svd.qi

    # Items
    idx_item = []
    for i in range(trainset.n_items):
        idx_item.append(trainset.to_raw_iid(i))
    idx_item_int = [int(item) for item in idx_item]
    print("#items in train set: " + str(len(idx_item)))

    np.save(f"{PROJECT_DIR}/dataset/jester/idx_item", idx_item)

    action_features = pd.DataFrame(data=qi_all)
    action_features.insert(0, 'jokeid', idx_item_int)  # action_features["MovieID"] = idx
    action_features.to_csv(f"{PROJECT_DIR}/dataset/jester/action_features.csv", index=False)

    # Users
    idx_user = []
    for i in range(trainset.n_users):
        idx_user.append(trainset.to_raw_uid(i))
    idx_user_int = [int(item) for item in idx_user]
    print("#users in train set: " + str(len(idx_user)))

    user_features = pd.DataFrame(data=pu_all)
    user_features.insert(0, 'userid', idx_user_int)
    user_features.to_csv(f"{PROJECT_DIR}/dataset/jester/user_features.csv", index=False)

    user_stream = make_user_stream(user_features, 15000)
    np.save(f"{PROJECT_DIR}/dataset/jester/user_stream", user_stream)

    # Ratings
    ratings = jokes.raw_ratings
    all_ratings = pd.DataFrame(data=ratings)
    all_ratings.columns = ['UserID', 'JokeID', 'Rating', 'other']
    del all_ratings['other']  # all_ratings.drop('other', axis=0)

    actions_all = all_ratings.groupby('JokeID').size().sort_values(ascending=False)  # [:140]

    actions_all = list(actions_all.index)
    # len(actions_all)
    top_jokes = all_ratings[all_ratings['JokeID'].isin(actions_all)]

    top_jokes['Reward'] = np.where(top_jokes['Rating'] >= 0, 1, 0)  # if rating >=0, the user will like the joke

    top_jokes["JokeID"] = pd.to_numeric(top_jokes["JokeID"])
    top_jokes["UserID"] = pd.to_numeric(top_jokes["UserID"])
    top_jokes.to_csv(f"{PROJECT_DIR}/dataset/jester/top_jokes.csv")

    reward_list = top_jokes[['UserID', 'JokeID', 'Reward']]
    reward_list = reward_list[reward_list['Reward'] == 1]
    reward_list.to_csv(f"{PROJECT_DIR}/dataset/jester/reward_list.csv")


if __name__ == '__main__':
    main_data()