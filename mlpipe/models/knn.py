from sklearn.neighbors import KNeighborsClassifier
import numpy as np

try:
    import pickle
except ImportError:
    import cPickle as pickle

from mlpipe import Model


class KNNModel(Model):


    def __init__(self, name="KNNModel", features=[], **kwargs):
        Model.__init__(self)
        self.name = name
        if len(features) == 0:
            self.features = ['corrLive', 'rmsLive', 'kurtLive', 'DELive',
                             'MFELive', 'skewLive', 'normLive',
                             'jumpLive', 'gainLive',
                             'psel', 'resp', 'respSel', 'cal',
                             'ff','stable','alt','pwv']
        else:
            self.features = features
        self.model = KNeighborsClassifier(**kwargs)

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
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
