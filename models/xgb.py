from xgboost import XGBClassifier
import numpy as np
import cPickle as pickle

from mlpipe import Model


class XGBModel(Model):

    name = "XGBoost"

    def __init__(self):
        Model.__init__(self)
        self.model = XGBClassifier()
        self.features = ['corrLive', 'rmsLive', 'kurtLive', 'DELive',
                         'MFELive', 'skewLive', 'normLive', 'darkRatioLive',
                         'jumpLive', 'gainLive']

    def train(self, data, labels, metadata):
        # gather all metadata to form the features
        features = np.hstack([metadata[key] for key in self.features])
        self.model.fit(features, labels)
    
    def validate(self, data, labels, metadata):
        # gather all metadata to form the features
        features = np.hstack([metadata[key] for key in self.features])
        prediction = self.model.predict(features)
        prediction_proba = self.model.predict_proba(features)
        return prediction, prediction_proba

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)

