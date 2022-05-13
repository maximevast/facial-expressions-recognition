#!/usr/bin/env python3

import sys
import collections
import time

from sklearn.neural_network   import MLPClassifier
from sklearn.svm              import SVC
from sklearn.metrics          import classification_report, confusion_matrix

from utils       import Frame, Point
from classifiers import SklearnClassifier


algorithms = collections.OrderedDict([
    ('MLP', MLPClassifier(solver='sgd', alpha=0.1, hidden_layer_sizes=(300,), random_state=0, activation='relu', max_iter=2000, learning_rate='adaptive')),
    ('SVC', SVC()),
])

for algorithm_name, algorithm_obj in algorithms.items():
    start = time.time()

    classifier = SklearnClassifier()
    classifier.load_from_file(sys.argv[1], 'topics')
    classifier.set_data()
    # uncomment the following line to train and test on the same user
    # classifier.set_data(training='a', testing='a')
    classifier.scale()
    classifier.use(algorithm_obj)
    classifier.fit()
    score  = classifier.score(classifier.test_features, classifier.test_classes)
    classes_pred = classifier.predict(classifier.test_features)

    print(algorithm_name + ' : ' + str(score))
    print('Elapsed time to train and test : ' + str(time.time() - start))
    print(classification_report(classifier.test_classes, classes_pred))
    print(confusion_matrix(classifier.test_classes, classes_pred))
