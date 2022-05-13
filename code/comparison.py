#!/usr/bin/env python3

import sys
import collections

from sklearn.linear_model     import LogisticRegressionCV, SGDClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC, LinearSVC
from sklearn.naive_bayes      import GaussianNB, BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier

from utils       import Frame, Point
from classifiers import SklearnClassifier

files  = [
    'affirmative',
    'conditional',
    'doubt_question',
    'emphasis',
    'negative',
    'relative',
    'topics',
    'wh_question',
    'yn_question'
]
algorithms = collections.OrderedDict([
    ('MLP', MLPClassifier(solver='sgd', alpha=0.1, hidden_layer_sizes=(300,), random_state=0, activation='relu', max_iter=2000, learning_rate='adaptive')),
    ('LogisticRegCV', LogisticRegressionCV()),
    ('KNN', KNeighborsClassifier()),
    ('SVC', SVC()),
    ('LinearSVC', LinearSVC()),
    ('GaussianNB', GaussianNB()),
    ('BernoulliNB', BernoulliNB()),
    ('SGD', SGDClassifier()),
    ('GaussianProcess', GaussianProcessClassifier())
])
output = {}

for f in files:
    output[f] = {}

    for algorithm_name, algorithm_obj in algorithms.items():
        classifier = SklearnClassifier()
        classifier.load_from_file(sys.argv[1], f)
        classifier.set_data()
        # uncomment the following line to train and test on the same user
        # classifier.set_data(training='b', testing='b')
        classifier.scale()
        classifier.use(algorithm_obj)
        classifier.fit()
        output[f][algorithm_name] = classifier.score(classifier.test_features, classifier.test_classes)


arrayHeader = "\033[94m{:<24}\033[0m".format('Sentence Type')

for algorithm_name, algorithm_obj in algorithms.items():
    arrayHeader += "\033[94m{:<16}\033[0m".format(algorithm_name)

print(arrayHeader)

for sentence_type in output:
    print('{:<18}'.format(sentence_type), end='\t', flush=True)

    for algorithm in algorithms:
        value = round(output[sentence_type][algorithm], 4)

        if value >= 0.8:
            value = '\033[91m ' + str(value) + '\033[0m'
        elif value >= 0.7:
            value = '\033[93m ' + str(value) + '\033[0m'
        else:
            value = ' ' + str(value)

        print(value, end='\t\t', flush=True)
    print()
