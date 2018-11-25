from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cPickle as pickle

from mlpipe import Model


class KNNModel(Model):

    name = "KNNModel"

    def __init__(self):
        Model.__init__(self)
        
    def setup(self, device):
        self.knn = KNeighborsClassifier(n_neighbors=7)

    def train(self, data, labels, metadata):
        corrLive = metadata['corrLive'][:,None]
        rmsLive = metadata['rmsLive'][:,None]
        kurtLive = metadata['kurtLive'][:,None]
        skewLive = metadata['skewLive'][:,None]
        normLive = metadata['normLive'][:,None]
        features = np.hstack([corrLive, rmsLive, kurtLive, skewLive, normLive])
        self.knn.fit(features, labels)
    
    def validate(self, data, labels, metadata):
        corrLive = metadata['corrLive'][:,None]
        rmsLive = metadata['rmsLive'][:,None]
        kurtLive = metadata['kurtLive'][:,None]
        skewLive = metadata['skewLive'][:,None]
        normLive = metadata['normLive'][:,None]
        features = np.hstack([corrLive, rmsLive, kurtLive, skewLive, normLive])
        prediction = self.knn.predict(features)
        return prediction

    def save(self, filename):
        pickle.dump(self.knn, filename)

