#!/usr/bin/env python3

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import Frame, Point, build_data_set


class Classifier:
    def load_from_file(self, directory, class_file):

        user_a_file = directory + '/a_' + class_file
        user_b_file = directory + '/b_' + class_file

        user_a_data = build_data_set(user_a_file)
        user_b_data = build_data_set(user_b_file)

        self.data = {
            'a': user_a_data,
            'b': user_b_data
        }

    def set_data(self, training='a', testing='b'):
        X_train = np.array([f.flatten() for f in self.data[training]])
        y_train  = np.array([f.binary_class for f in self.data[training]])
        X_test  = np.array([f.flatten() for f in self.data[testing]])
        y_test   = np.array([f.binary_class for f in self.data[testing]])

        if training == testing:
            self.train_features, self.test_features, self.train_classes, self.test_classes = train_test_split(X_train, y_train, test_size=0.1)
        else:
            self.train_features = X_train
            self.train_classes  = y_train
            self.test_features  = X_test
            self.test_classes   = y_test

    def scale(self):
        train_scaler = MinMaxScaler()
        test_scaler  = MinMaxScaler()
        self.train_features  = train_scaler.fit_transform(self.train_features)
        self.test_features   = test_scaler.fit_transform(self.test_features)


class SklearnClassifier(Classifier):

    def use(self, algorithm):
        self.algorithm  = algorithm

    def fit(self, features=None, classes=None):
        if features is None:
            features = self.train_features

        if classes is None:
            classes = self.train_classes

        return self.algorithm.fit(features, classes)

    def score(self, features, classes):
        return self.algorithm.score(features, classes)

    def predict(self, features):
        return self.algorithm.predict(features)
