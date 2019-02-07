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

    def train(self, data, labels, metadata):
        corrLive = metadata['corrLive']
        rmsLive = metadata['rmsLive']
        kurtLive = metadata['kurtLive']
        DELive = metadata['DELive']
        MFELive = metadata['MFELive']
        skewLive = metadata['skewLive']
        normLive = metadata['normLive']
        darkRatioLive = metadata['darkRatioLive']
        jumpLive = metadata['jumpLive']
        gainLive = metadata['gainLive']
        features = np.hstack([corrLive, rmsLive, kurtLive, DELive,
                              MFELive, skewLive, normLive, darkRatioLive, jumpLive,
                              gainLive])
        self.model.fit(features, labels)
    
    def validate(self, data, labels, metadata):
        corrLive = metadata['corrLive']
        rmsLive = metadata['rmsLive']
        kurtLive = metadata['kurtLive']
        DELive = metadata['DELive']
        MFELive = metadata['MFELive']
        skewLive = metadata['skewLive']
        normLive = metadata['normLive']
        darkRatioLive = metadata['darkRatioLive']
        jumpLive = metadata['jumpLive']
        gainLive = metadata['gainLive']
        features = np.hstack([corrLive, rmsLive, kurtLive, DELive,
                              MFELive, skewLive, normLive, darkRatioLive, jumpLive,
                              gainLive])
        prediction = self.model.predict(features)
        return prediction

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

