from datetime import datetime
import numpy as np
import pandas as pd


class RecommenderDataset:
    def __init__(
        self,
        actions,
        action_features,
        action_biases,
        user_stream,
        true_user_features,
        user_features,
        user_biases,
        reward_list,
        ratings_list,
        ratings_range,
        implicit_feedback,  # If implicit feedback estimate unseen ratings as 0 for NDCG.
    ):
        self.actions: list = actions
        self.action_features: pd.DataFrame = action_features
        self.action_biases: pd.DataFrame = action_biases

        self.user_stream: pd.DataFrame = user_stream
        # User features present in the data. Used in the MovieLens experiment.
        self.true_user_features: pd.DataFrame = true_user_features
        # User features extracted by the SVD.
        self.user_features: pd.DataFrame = user_features
        self.user_biases: pd.DataFrame = user_biases

        self.reward_list: pd.DataFrame = reward_list
        self.ratings_list: pd.DataFrame = ratings_list
        self.ratings_range: tuple = ratings_range
        self.implicit_feedback: bool = implicit_feedback

        (
            self.action_context_dict,
            self.action_bias_dict,
            self.true_user_features_array,
            self.user_features_array,
            self.user_biases_array,
            self.user_id_to_index,
            self.watched_list_series,
            self.user_id_to_watched_list_index,
            self.user_item_to_rating,
        ) = self.data_preprocessing()

    def data_preprocessing(self):
        """Preprocess the data to be in a format that allows more optimal policy evaluation.
        This method reindexes true_user_features, user_features, user_biases and reward_list.
        """
        action_context_dict = {}
        action_bias_dict = {}
        for action_id in self.actions:
            action_context = np.array(
                self.action_features[self.action_features["item_id"] == action_id]
            )[0][1:]
            action_bias = self.action_biases[
                self.action_biases["item_id"] == action_id
            ].iloc[0][1]
            action_context_dict[action_id] = action_context.astype(np.float64)
            action_bias_dict[action_id] = action_bias

        if "user_id" in self.true_user_features.columns:
            self.true_user_features = self.true_user_features.set_index("user_id")
        true_user_features_array = self.true_user_features.to_numpy(dtype=np.float32)

        self.user_features = self.user_features.set_index("user_id")
        self.user_features = self.user_features.reindex(self.true_user_features.index)
        user_features_array = self.user_features.to_numpy(dtype=np.float32)

        assert (
            self.user_features.index == self.true_user_features.index
        ).all(), "Indexes of user features and true u.f. dont coincide"

        self.user_biases = self.user_biases.set_index("user_id")
        self.user_biases = self.user_biases.reindex(self.user_features.index)
        user_biases_array = self.user_biases.to_numpy(dtype=np.float32)

        assert (
            self.user_features.index == self.user_biases.index
        ).all(), "Indexes of user features and biases dont coincide"

        # This is an optimization because pandas indexing is slower.
        user_id_to_index = {
            uid: ind for ind, uid in enumerate(self.user_features.index)
        }

        self.reward_list = self.reward_list.set_index("user_id")
        watched_list_series = (
            self.reward_list.groupby("user_id")["item_id"].agg(set=set).set
        )
        user_id_to_watched_list_index = {
            uid: ind for ind, uid in enumerate(watched_list_series.index)
        }

        user_item_to_rating = {}
        for (user_id, item_id, rating) in self.ratings_list.to_numpy():
            user_item_to_rating[(user_id, item_id)] = rating

        return (
            action_context_dict,
            action_bias_dict,
            true_user_features_array,
            user_features_array,
            user_biases_array,
            user_id_to_index,
            watched_list_series,
            user_id_to_watched_list_index,
            user_item_to_rating,
        )

    # TODO Remove this function and everything that uses it.
    def get_full_data(self):
        """For legacy compatibility reasons"""
        return (
            self.actions,
            self.action_features,
            self.action_biases,
            self.user_stream,
            self.true_user_features,
            self.user_features,
            self.user_biases,
            self.reward_list,
            self.ratings_list,
        )

    def get_full_context(self, user_t):
        user_feature_index = self.user_id_to_index[user_t]
        true_user_feature = self.true_user_features_array[user_feature_index]

        full_context = {}
        for i, action_id in enumerate(self.actions):
            action_context = self.action_context_dict[action_id]
            full_context[action_id] = np.append(action_context, true_user_feature)

        return full_context

    def get_score_true(self, user_t):
        user_feature_index = self.user_id_to_index[user_t]

        user_feature = self.user_features_array[user_feature_index]
        user_bias = self.user_biases_array[user_feature_index]
        # True ratings used to compute the NDCG score.
        score_true = np.zeros((1, len(self.actions)))
        for i, action_id in enumerate(self.actions):
            action_context = self.action_context_dict[action_id]
            action_bias = self.action_bias_dict[action_id]

            user_item_tuple = (user_t, action_id)
            # If item is rated by user - use true rating as score.
            if user_item_tuple in self.user_item_to_rating:
                rating = self.user_item_to_rating[user_item_tuple]
            else:
                if self.implicit_feedback:
                    rating = 0
                else:
                    # Otherwise use rating estimated by matrix factorization.
                    # See https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD
                    rating = (
                        user_bias + action_bias + np.dot(action_context, user_feature)
                    )
                    rating = (
                        rating
                        if rating > self.ratings_range[0]
                        else self.ratings_range[0]
                    )
                    rating = (
                        rating
                        if rating < self.ratings_range[1]
                        else self.ratings_range[1]
                    )

            score_true[:, i] = rating

        return score_true

    def generate_users(self, time_steps):
        """Generate users and their corresponding data."""
        # Use last `time_steps` users in the user stream. We do this because later data is more dense, so we get all
        # users from a smaller time window this way.
        assert (
            time_steps <= self.user_stream.shape[0]
        ), "Not enough users in user stream for given --times parameter"
        ind_start = self.user_stream.shape[0] - time_steps
        ind_end = self.user_stream.shape[0] - 1
        if "timestamp" in self.user_stream.columns:
            print(
                f"First user in exp from {datetime.fromtimestamp(self.user_stream.timestamp[ind_start])}"
            )
            print(
                f"Last user in exp from {datetime.fromtimestamp(self.user_stream.timestamp[ind_end])}"
            )

        t = 0
        user_ind = ind_start
        while t < ind_end - ind_start:
            user_ind += 1
            user_t = self.user_stream.iloc[user_ind, 0]

            try:
                watched_list_index = self.user_id_to_watched_list_index[user_t]
                watched_list = self.watched_list_series.iloc[watched_list_index]
            except KeyError:
                watched_list = set()

            # Create full context by concatenating user and item features.
            full_context = self.get_full_context(user_t)
            score_true = self.get_score_true(user_t)

            yield full_context, watched_list, score_true
            t += 1
