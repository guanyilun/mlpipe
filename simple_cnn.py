from mlpipe import MLPipe
from models.simple_cnn import CNNModel
from models.knn import KNNModel

pipe = MLPipe()
pipe.set_epochs(100)
pipe.set_dataset('data/dataset.h5')
pipe.add_model(CNNModel())
pipe.add_model(KNNModel())
pipe.run()
