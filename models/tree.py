"""
A simple SVC model, for reference please see
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

I am only using five pickle parameters as feaures, in principle
more features can be used and one can also generate features on the
go using the data passed in to the Model. 
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

try:
    import pickle
except ImportError:
    import cPickle as pickle

from mlpipe import Model


class DecisionTreeModel(Model):

    name = "DecisionTree"

    def __init__(self, random_state=0):
        self.model = DecisionTreeClassifier(random_state=random_state)
        self.features = ['corrLive', 'rmsLive', 'kurtLive', 'DELive',
                         'MFELive', 'skewLive', 'normLive', 'darkRatioLive',
                         'jumpLive', 'gainLive', 'feat1', 'feat2', 'feat3']

    def train(self, data, labels, metadata):
        features = np.hstack([metadata[key] for key in self.features])
        self.model.fit(features, labels)

    def validate(self, data, labels, metadata):
        features = np.hstack([metadata[key] for key in self.features])
        prediction = self.model.predict(features)
        prediction_proba = self.model.predict_proba(features)
        return prediction, prediction_proba
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
