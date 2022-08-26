import numpy as np
import pandas as pd
import random
import surprise as sp

from config.cofig import PROJECT_DIR, AMAZON_CONTEXT_DIMENSION


random.seed(42)


AMAZON_RATINGS_PATH = f"{PROJECT_DIR}/dataset/amazon/Video_Games.csv"
AMAZON_NUMBER_OF_ACTIONS = 100


def create_actions_users_and_rewards(ratings_df):
    """Preprocess the original dataframe to extract actions, users and rewards from it.

    1. Extract top AMAZON_NUMBER_OF_ACTIONS items from the data (sort by number of ratings).
    2. Then extract users which rated those extracted items. This way, for every user at least one arm
    will have reward 1 later.
    3. Create rewards from ratings by considering rating > 3.0 being reward 1. Then only save user-item-reward triplets
    for rewards equal to 1. This way we can filter by user during evaluation and if item is there reward is 1.
    """
    actions = ratings_df.groupby("item_id").size().sort_values(ascending=False)[:AMAZON_NUMBER_OF_ACTIONS]
    actions = list(actions.index)

    # Only consider users that have watched some movies from the considered actions.
    top_ratings = ratings_df[ratings_df["item_id"].isin(actions)]
    top_ratings = top_ratings.sort_values("timestamp", ascending=1)
    user_stream = top_ratings["user_id"]

    unique_users = user_stream.unique()
    print(f"Experiments has {len(actions)} items,\n{len(user_stream)} users\nof which {len(unique_users)} are unique.")

    top_ratings["reward"] = np.where(top_ratings["rating"] >= 3.0, 1, 0)
    reward_list = top_ratings[["user_id", "item_id", "reward"]]
    reward_list = reward_list[reward_list['reward'] == 1]
    return actions, user_stream, reward_list


def preprocess_amazon_data():
    """Use surprise library to create user and item features and rewards from the original amazon ratings data."""
    ratings_df = pd.read_csv(AMAZON_RATINGS_PATH, names=["item_id", "user_id", "rating", "timestamp"])

    actions, user_stream, reward_list = create_actions_users_and_rewards(ratings_df)

    user_stream.to_csv(f"{PROJECT_DIR}/dataset/amazon/user_stream.csv", sep='\t', index=False)
    pd.DataFrame(actions, columns=["item_id"]).to_csv(f"{PROJECT_DIR}/dataset/amazon/actions.csv", sep='\t', index=False)
    reward_list.to_csv(f"{PROJECT_DIR}/dataset/amazon/reward_list.csv", sep='\t', index=False)

    # Use SVD to create user and item features.
    full_dataset = sp.Dataset.load_from_df(
        # order of columns matters here!
        ratings_df[["user_id", "item_id", "rating"]], reader=sp.Reader(rating_scale=(1.0, 5.0))
    )
    trainset, testset = sp.model_selection.train_test_split(full_dataset, test_size=0.1)

    print(f"User-rating matrix filled to total ratio: {trainset.n_ratings / (trainset.n_items * trainset.n_users)}")

    # Dividing the context size by half because user and item context will be concatenated during evaluation.
    svd = sp.SVD(n_factors=AMAZON_CONTEXT_DIMENSION // 2, n_epochs=10)
    svd.fit(trainset)

    predictions = svd.test(testset)
    print(f"RMSE of SVD: {sp.accuracy.rmse(predictions)}")

    # SVD returns memory-views (cython), hence asarray calls.
    pu_all, qi_all = np.asarray(svd.pu), np.asarray(svd.qi)

    trainset_item_ids = [trainset.to_raw_iid(i) for i in range(trainset.n_items)]
    items_that_have_features_set = set(trainset_item_ids)
    items_without_features_counter = sum([1 for item_id in actions if item_id not in items_that_have_features_set])
    print(f"Items without features: {items_without_features_counter}")

    action_features = pd.DataFrame(data=qi_all)
    action_features.insert(loc=0, column='item_id', value=trainset_item_ids)
    action_features.to_csv(f"{PROJECT_DIR}/dataset/amazon/action_features.csv", index=False)

    trainset_user_ids = [trainset.to_raw_uid(i) for i in range(trainset.n_users)]
    users_that_have_features_set = set(trainset_user_ids)
    users_without_features_counter = sum(
        [1 for user_id in user_stream.unique() if user_id not in users_that_have_features_set]
    )
    print(f"Users without features: {users_without_features_counter}")

    user_features = pd.DataFrame(data=pu_all)
    user_features.insert(loc=0, column='user_id', value=trainset_user_ids)
    user_features.to_csv(f"{PROJECT_DIR}/dataset/amazon/user_features.csv", index=False)


if __name__ == "__main__":
    preprocess_amazon_data()