"""
A simple SVC model, for reference please see
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

I am only using five pickle parameters as feaures, in principle
more features can be used and one can also generate features on the
go using the data passed in to the Model.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import pickle
except ImportError:
    import cPickle as pickle

from mlpipe import Model


class RFModel(Model):
    def __init__(self, name="RandomForest", features=[], **kwargs):
        self.name = name
        self.model = RandomForestClassifier(**kwargs)
        if len(features) == 0:
            self.features = ['corrLive', 'rmsLive', 'kurtLive', 'DELive',
                             'MFELive', 'skewLive', 'normLive',
                             'jumpLive', 'gainLive',
                             'psel', 'resp', 'respSel', 'cal',
                             'ff','stable','alt','pwv']
        else:
            self.features = features

    def train(self, data, labels, metadata):
        features = np.hstack([metadata[key] for key in self.features])
        self.model.fit(features, labels)

    def validate(self, data, labels, metadata):
        features = np.hstack([metadata[key] for key in self.features])
        prediction = self.model.predict(features)
        prediction_prob = self.model.predict_proba(features)
        return prediction, prediction_prob

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, features):
        prediction = self.model.predict(features)
        return prediction
