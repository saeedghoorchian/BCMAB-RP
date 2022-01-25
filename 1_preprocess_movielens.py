import itertools
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from config.cofig import PROJECT_DIR



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

    # streaming_batch: the result for testing bandit algrorithms
    top50_data = data[data['movie_id'].isin(actions)]
    top50_data = top50_data.sort_values('timestamp', ascending=1)
    streaming_batch = top50_data['user_id']

    # reward_list: if rating >=3, the user will watch the movie
    top50_data['reward'] = np.where(top50_data['rating'] >= 3, 1, 0)
    reward_list = top50_data[['user_id', 'movie_id', 'reward']]
    reward_list = reward_list[reward_list['reward'] == 1]
    return streaming_batch, user_feature, actions, reward_list


def main_data():
    # read and preprocess the movie data
    movie = pd.read_table(f'{PROJECT_DIR}/dataset/movielens/movies.dat', sep='::', names=['movie_id', 'movie_name', 'tag'], engine='python')
    movie = movie_preprocessing(movie)

    # read the ratings data and merge it with movie data
    rating = pd.read_table(f"{PROJECT_DIR}/dataset/movielens/ratings.dat", sep="::",
                           names=["user_id", "movie_id", "rating", "timestamp"], engine='python')
    data = pd.merge(rating, movie, on="movie_id")

    # extract feature from our data set
    streaming_batch, user_feature, actions, reward_list = feature_extraction(data)
    streaming_batch.to_csv(f"{PROJECT_DIR}/dataset/movielens/streaming_batch.csv", sep='\t', index=False)
    user_feature.to_csv(f"{PROJECT_DIR}/dataset/movielens/user_feature.csv", sep='\t')
    pd.DataFrame(actions, columns=['movie_id']).to_csv(f"{PROJECT_DIR}/dataset/movielens/actions.csv", sep='\t', index=False)
    reward_list.to_csv(f"{PROJECT_DIR}/dataset/movielens/reward_list.csv", sep='\t', index=False)

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
    algorithm = SVD(n_factors=100)
    algorithm.fit(trainset)

    idx = []
    for i in range(trainset.n_items):
        idx.append(trainset.to_raw_iid(i))

    qi_all = algorithm.qi

    action_features = pd.DataFrame(data=qi_all)
    action_features.insert(0, 'movieid', idx)  # action_features["MovieID"] = idx
    action_features.to_csv(f"{PROJECT_DIR}/dataset/movielens/action_features_2_CBRAP_BCMABRP.csv", index=False)


if __name__ == '__main__':
    main_data()