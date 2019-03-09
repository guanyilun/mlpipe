from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cPickle as pickle

from mlpipe import Model


class KNNModel(Model):

    name = "KNNModel"

    def __init__(self, n_neighbors=7):
        Model.__init__(self)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.name = '{}-{}'.format(self.name, n_neighbors)
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
