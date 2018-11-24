from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from mlpipe import Model

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)


class KNNModel(Model):

    name = "KNNModel"

    def __init__(self):
        Model.__init__(self)
        
    def setup(self, device):
        self.knn = KNeighborsClassifier(n_neighbors=7)

    def train(self, data, labels, metadata):
        corrLive = metadata['corrLive']
        rmsLive = metadata['rmsLive']
        kurtLive = metadata['kurtLive']
        skewLive = metadata['skewLive']
        normLive = metadata['normLive']
        features = np.hstack([corrLive, rmsLive, kurtLive, skewLive, normLive])
        self.knn.fit(features, labels)
        prediction = knn.predict(features)
        return prediction
    
    def test(self, data, labels, metadata):
        pass
