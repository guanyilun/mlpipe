from xgboost import XGBClassifier
import numpy as np
import cPickle as pickle

from mlpipe import Model


class XGBModel(Model):

    name = "XGBoost"

    def __init__(self):
        Model.__init__(self)
        self.model = XGBClassifier()

    def train(self, data, labels, metadata):
        corrLive = metadata['corrLive']
        rmsLive = metadata['rmsLive']
        kurtLive = metadata['kurtLive']
        skewLive = metadata['skewLive']
        normLive = metadata['normLive']
        features = np.hstack([corrLive, rmsLive, kurtLive, skewLive, normLive])
        self.model.fit(features, labels)
    
    def validate(self, data, labels, metadata):
        corrLive = metadata['corrLive']
        rmsLive = metadata['rmsLive']
        kurtLive = metadata['kurtLive']
        skewLive = metadata['skewLive']
        normLive = metadata['normLive']
        features = np.hstack([corrLive, rmsLive, kurtLive, skewLive, normLive])
        prediction = self.model.predict(features)
        return prediction

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

