import pandas as pd
from config.cofig import PROJECT_DIR


def get_movielens_data():
    # streaming_batch = pd.read_csv("streaming_batch.csv", sep="\t", names=["user_id"], engine="c")

    actions_id = list(pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/actions.csv", sep="\t", header=0, engine="c")["item_id"])
    action_features = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/action_features.csv")
    action_biases = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/action_biases.csv")

    user_stream = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/user_stream.csv", sep="\t", header=0, engine="c")
    true_user_features = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/true_user_features.csv", sep="\t", header=0, index_col=0, engine="c")
    user_features = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/user_features.csv")
    user_biases = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/user_biases.csv")

    reward_list = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/reward_list.csv", sep="\t", header=0, engine="c")
    ratings_list = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/ratings_list.csv", sep="\t", header=0, engine="c")
    # action_context = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/action_context.csv", sep="\t", header=0, engine="c")

    movie = pd.read_csv(f"{PROJECT_DIR}/dataset/movielens/movie.csv", sep="\t", header=0, engine="c")
    # actions = actions_id #[]
    # for key in actions_id:
    # print(key)
    # action = Action(key) #?
    # actions.append(action)
    return actions_id, action_features, action_biases, user_stream, true_user_features, user_features, user_biases, reward_list, ratings_list,  # , movie