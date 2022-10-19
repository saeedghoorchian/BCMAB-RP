import itertools
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from config.cofig import PROJECT_DIR

THRESHOLD = None


def movie_preprocessing(movie):
    movie_col = list(movie.columns)
    movie_tag = [doc.split('|') for doc in movie['tag']]
    tag_table = {token: idx for idx, token in enumerate(set(itertools.chain.from_iterable(movie_tag)))}
    movie_tag = pd.DataFrame(movie_tag)
    tag_table = pd.DataFrame(tag_table.items())
    tag_table.columns = ['Tag', 'Index']

    # use one-hot encoding for movie genres (here called tag)
    tag_dummy = np.zeros([len(movie), len(tag_table)])

    for i in range(len(movie)):
        for j in range(len(tag_table)):
            if tag_table['Tag'][j] in list(movie_tag.iloc[i, :]):
                tag_dummy[i, j] = 1

    # combine the tag_dummy one-hot encoding table to original movie files
    movie = pd.concat([movie, pd.DataFrame(tag_dummy)], 1)
    movie_col.extend(['tag' + str(i) for i in range(len(tag_table))])
    movie.columns = movie_col
    movie = movie.drop('tag', 1)
    return movie


def feature_extraction(data):
    # actions: we use top 1000 movies as our actions for recommendations
    actions = data.groupby('movie_id').size().sort_values(ascending=False)[:1000]
    actions = list(actions.index)

    # user_feature: tags they've watched for non-top-1000 movies normalized per user
    user_feature = data[~data['movie_id'].isin(actions)]
    user_feature = user_feature.groupby('user_id').aggregate(np.sum)
    user_feature = user_feature.drop(['movie_id', 'rating', 'timestamp'], 1)
    user_feature = user_feature.div(user_feature.sum(axis=1), axis=0)

    # user_stream: the result for testing bandit algrorithms
    # Only consider users that have watched some movies from the considered actions.
    top50_data = data[data['movie_id'].isin(actions)]
    top50_data = top50_data.sort_values('timestamp', ascending=1)
    user_stream = top50_data[["user_id", "timestamp"]]
    # Only use users with features.
    user_stream = user_stream[user_stream.user_id.isin(set(user_feature.index))]

    users_all_exp = user_stream.user_id[:100000].unique()
    print(f"---\nThere are {len(users_all_exp)} unique users in the experiment\n---")

    # reward_list: if rating >=3, the user will watch the movie
    if THRESHOLD is not None:
        top50_data['reward'] = np.where(top50_data['rating'] >= THRESHOLD, 1, 0)
    else:
        top50_data['reward'] = np.where(top50_data['rating'] >= 3, 1, 0)
    top50_data = top50_data.rename(columns={'movie_id': "item_id"})
    reward_list = top50_data[['user_id', 'item_id', 'reward']]
    reward_list = reward_list[reward_list['reward'] == 1]

    ratings_list = top50_data[["user_id", "item_id", "rating"]]
    return user_stream, user_feature, actions, reward_list, ratings_list


def main_data():
    # read and preprocess the movie data
    movie = pd.read_table(f'{PROJECT_DIR}/dataset/movielens/movies.dat', sep='::', names=['movie_id', 'movie_name', 'tag'], engine='python')
    movie = movie_preprocessing(movie)

    # read the ratings data and merge it with movie data
    rating = pd.read_table(f"{PROJECT_DIR}/dataset/movielens/ratings.dat", sep="::",
                           names=["user_id", "movie_id", "rating", "timestamp"], engine='python')
    data = pd.merge(rating, movie, on="movie_id")

    # extract feature from our data set
    user_stream, true_user_features, actions, reward_list, ratings_list = feature_extraction(data)
    true_user_features.to_csv(f"{PROJECT_DIR}/dataset/movielens/true_user_features.csv", sep='\t')
    pd.DataFrame(actions, columns=['item_id']).to_csv(f"{PROJECT_DIR}/dataset/movielens/actions.csv", sep='\t', index=False)
    if THRESHOLD is not None:
        reward_list.to_csv(f"{PROJECT_DIR}/dataset/movielens/reward_list_{THRESHOLD}.csv", sep='\t', index=False)
    else:
        reward_list.to_csv(f"{PROJECT_DIR}/dataset/movielens/reward_list.csv", sep='\t', index=False)
    ratings_list.to_csv(f"{PROJECT_DIR}/dataset/movielens/ratings_list.csv", sep='\t', index=False)

    action_context = movie[movie['movie_id'].isin(actions)]
    action_context.to_csv(f"{PROJECT_DIR}/dataset/movielens/action_context.csv", sep='\t', index = False)

    movie.to_csv(f"{PROJECT_DIR}/dataset/movielens/movie.csv", sep='\t', index=False)

    ratings_df = pd.read_table(f"{PROJECT_DIR}/dataset/movielens/ratings.dat", sep="::",
                               names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
    ratings_df['Rating'].unique()

    # instantiate a reader and read in our rating data
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)

    # train SVD on 75% of known rates
    trainset, testset = train_test_split(data, test_size=.25)
    svd = SVD(n_factors=100)
    svd.fit(trainset)

    idx_item = []
    for i in range(trainset.n_items):
        idx_item.append(trainset.to_raw_iid(i))

    # SVD returns memory-views (cython), hence asarray calls.
    pu_all, qi_all = np.asarray(svd.pu), np.asarray(svd.qi)
    bu_all, bi_all = np.asarray(svd.bu), np.asarray(svd.bi)

    action_features = pd.DataFrame(data=qi_all)
    action_features.insert(loc=0, column='item_id', value=idx_item)  # action_features["MovieID"] = idx
    action_features.to_csv(f"{PROJECT_DIR}/dataset/movielens/action_features.csv", index=False)

    action_biases = pd.DataFrame(data=bi_all)
    action_biases.insert(loc=0, column='item_id', value=idx_item)
    action_biases.to_csv(f"{PROJECT_DIR}/dataset/movielens/action_biases.csv", index=False)

    # Users
    idx_user = []
    for i in range(trainset.n_users):
        idx_user.append(trainset.to_raw_uid(i))
    idx_user_int = [int(item) for item in idx_user]
    print(f"#users in train set: {len(idx_user)}")

    user_features = pd.DataFrame(data=pu_all)
    user_features.insert(0, 'user_id', idx_user_int)

    # Only save user features for those users that are present in the experiment.
    user_features = user_features[user_features.user_id.isin(set(user_stream["user_id"]))]
    user_features.to_csv(f"{PROJECT_DIR}/dataset/movielens/user_features.csv", index=False)

    users_that_have_features_set = set(idx_user)
    # Only save users with features for the experiment.
    user_stream = user_stream[user_stream.user_id.isin(users_that_have_features_set)]

    user_stream.to_csv(f"{PROJECT_DIR}/dataset/movielens/user_stream.csv", sep='\t', index=False)

    user_biases = pd.DataFrame(data=bu_all)
    user_biases.insert(loc=0, column='user_id', value=idx_user_int)
    # Only save user biases for those users that are present in the experiment.
    user_biases = user_biases[user_biases.user_id.isin(set(user_stream["user_id"]))]
    user_biases.to_csv(f"{PROJECT_DIR}/dataset/movielens/user_biases.csv", index=False)


if __name__ == '__main__':
    # for t in list(map(lambda x: x / 2, range(1, 11))):
    #     THRESHOLD = t
    main_data()