import pandas as pd


def get_movielens_data():
    # streaming_batch = pd.read_csv('streaming_batch.csv', sep='\t', names=['user_id'], engine='c')
    streaming_batch = pd.read_csv('dataset/movielens/streaming_batch.csv', sep='\t', header=0, engine='c')
    user_feature = pd.read_csv('dataset/movielens/user_feature.csv', sep='\t', header=0, index_col=0, engine='c')
    actions_id = list(pd.read_csv('dataset/movielens/actions.csv', sep='\t', header=0, engine='c')['movie_id'])
    reward_list = pd.read_csv('dataset/movielens/reward_list.csv', sep='\t', header=0, engine='c')
    action_context = pd.read_csv('dataset/movielens/action_context.csv', sep='\t', header=0, engine='c')

    movie = pd.read_csv('dataset/movielens/movie.csv', sep='\t', header=0, engine='c')

    # actions = actions_id #[]
    # for key in actions_id:
    # print(key)
    # action = Action(key) #?
    # actions.append(action)
    return streaming_batch, user_feature, actions_id, reward_list, action_context  # , movie