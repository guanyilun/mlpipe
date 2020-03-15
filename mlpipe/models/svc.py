"""
A simple SVC model, for reference please see
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

I am only using five pickle parameters as feaures, in principle
more features can be used and one can also generate features on the
go using the data passed in to the Model.
"""

import numpy as np
from sklearn.svm import SVC

try:
    import pickle
except ImportError:
    import cPickle as pickle

from mlpipe import Model


class SVCModel(Model):

    name = "SVCModel"

    def __init__(self):
        self.model = SVC(gamma='auto')
        self.features = ['corrLive', 'rmsLive', 'kurtLive', 'DELive',
                         'MFELive', 'skewLive', 'normLive',
                         'jumpLive', 'gainLive']

    def train(self, data, labels, metadata):
        features = np.hstack([metadata[key] for key in self.features])
        self.model.fit(features, labels)

    def validate(self, data, labels, metadata):
        features = np.hstack([metadata[key] for key in self.features])
        prediction = self.model.predict(features)
        return prediction

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
