"""
A simple SVC model, for reference please see
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

I am only using five pickle parameters as feaures, in principle
more features can be used and one can also generate features on the
go using the data passed in to the Model. 
"""

import numpy as np
from sklearn.svm import SVC
import cPickle as pickle

from mlpipe import Model


class SVCModel(Model):

    name = "SVCModel"

    def __init__(self):
        self.model = SVC(gamma='auto')

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
