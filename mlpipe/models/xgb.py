from xgboost import XGBClassifier
import numpy as np

try:
    import pickle
except ImportError:
    import cPickle as pickle

from mlpipe import Model


class XGBModel(Model):
    def __init__(self, name="XGBoost", features=[], **kwargs):
        Model.__init__(self)
        self.name = name
        self.model = XGBClassifier(**kwargs)
        if len(features) == 0:
            self.features = ['corrLive', 'rmsLive', 'kurtLive', 'DELive',
                             'psel', 'MFELive', 'skewLive', 'normLive',
                             'jumpLive', 'gainLive',
                             'resp', 'respSel', 'cal',
                             'ff','stable','alt','pwv']
        else:
            self.features = features

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
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, features):
        prediction = self.model.predict(features)
        return prediction
