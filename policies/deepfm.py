from collections import deque
import six
import numpy as np

import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


class DeepFM_OnlinePolicy():
    def __init__(self, context_dimension, batch_size=1):
        self.name = f"DeepFM (batch_size={batch_size})"
        self.context_dimension = context_dimension
        self.batch_size = batch_size

        self.model = None

        # Save chosen context and action at time t
        self.context_t = None
        self.action_t = None
        # Save last `batch_size` pairs of context and reward to train the model.
        # self.context_label_memory = deque(maxlen=batch_size)

        # Save all the seen data to train the network
        self.context_label_memory = deque(maxlen=100000)
        # Keep track of trials
        self.trial = 0

    def update_memory(self, memory):  # (context_t, reward_t)
        self.context_label_memory.append(memory)

    def update_model_params(self):
        if self.trial % self.batch_size != 0:
            return

        if len(self.context_label_memory) < self.batch_size:
            # Don't train the model until at least `batch_size` observations in the buffer
            return

        dataset = np.zeros(shape=(len(self.context_label_memory), self.context_dimension))
        labels = np.zeros(shape=(len(self.context_label_memory),))
        for idx, (context_t, reward_t) in enumerate(self.context_label_memory):
            dataset[idx] = context_t
            labels[idx] = reward_t

        ### Training on gathered data

        sparse_features = []
        dense_features = [i for i in range(self.context_dimension)]

        #         dataset[sparse_features] = dataset[sparse_features].fillna('-1', )
        #         dataset[dense_features] = dataset[dense_features].fillna(0, )
        target = labels

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        #         for feat in sparse_features:
        #             lbe = LabelEncoder()
        #             dataset[feat] = lbe.fit_transform(dataset[feat])

        mms = MinMaxScaler(feature_range=(0, 1))
        dataset[:, dense_features] = mms.fit_transform(dataset[:, dense_features])

        # 2.count #unique features for each sparse field,and record dense feature field name

        fixlen_feature_columns = [SparseFeat(feat, len(np.unique(dataset[:, feat])))
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        train_model_input = {feat_id: dataset[:, feat_id] for feat_id in feature_names}

        # 4.Define Model,train,predict and evaluate
        if self.model is None:
            device = 'cpu'
            use_cuda = True
            if use_cuda and torch.cuda.is_available():
                print('cuda ready...')
                device = 'cuda:0'

            self.model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                                task='binary',
                                l2_reg_embedding=1e-5, device=device)

            self.model.compile("adagrad", "binary_crossentropy",
                               #                       metrics=["binary_crossentropy", "auc"], )
                               metrics=[], )

        self.model.fit(
            train_model_input,
            target,
            batch_size=256,  # Default value in this library
            epochs=10,
            verbose=2,
            validation_split=0.0
        )
        # Clear the memory
        # self.context_label_memory.clear()

    def get_score(self, context, trial):
        self.trial = trial
        action_ids = list(six.viewkeys(context))
        context_array = np.asarray([context[action_id] for action_id in action_ids])

        # preprocessing for prediction
        sparse_features = []
        dense_features = [i for i in range(self.context_dimension)]

        #         dataset[sparse_features] = dataset[sparse_features].fillna('-1', )
        #         dataset[dense_features] = dataset[dense_features].fillna(0, )

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        #         for feat in sparse_features:
        #             lbe = LabelEncoder()
        #             dataset[feat] = lbe.fit_transform(dataset[feat])

        mms = MinMaxScaler(feature_range=(0, 1))
        context_array[:, dense_features] = mms.fit_transform(context_array[:, dense_features])

        # 2.count #unique features for each sparse field,and record dense feature field name

        fixlen_feature_columns = [SparseFeat(feat, context_array[feat].nunique())
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        model_input = {feat_id: context_array[:, feat_id] for feat_id in feature_names}

        if self.model is None:
            #             estimated_ctr_array = np.ones(context_array.shape[1])
            estimated_ctr_array = np.random.uniform(low=0.0, high=1.0, size=context_array.shape[0])
        else:
            estimated_ctr_array = self.model.predict(model_input, batch_size=len(action_ids))

        score_dict = {}
        for action_id, ctr in zip(action_ids, estimated_ctr_array):
            score_dict[action_id] = float(ctr)

        return score_dict

    def get_action(self, context, trial):

        score = self.get_score(context, trial)
        recommendation_id = max(score, key=score.get)
        self.full_context_t = context
        self.action_t = recommendation_id
        return recommendation_id, score

    def reward(self, reward_t):
        context_t = self.full_context_t[self.action_t]
        self.update_memory((context_t, reward_t))
        self.update_model_params()
