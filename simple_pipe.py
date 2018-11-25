from mlpipe import MLPipe
from models.cnn import CNNModel
from models.knn import KNNModel

pipe = MLPipe()
pipe.set_epochs(1)
pipe.set_dataset('data/dataset.h5')

pipe.add_model(CNNModel())
pipe.add_model(KNNModel(7))
pipe.add_model(KNNModel(5))

pipe.train()
pipe.test()
pipe.save('saved_runs/1125/')
pipe.clean()
