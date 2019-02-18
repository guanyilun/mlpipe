"""
This script is the boilerplate for a simple pipeline where no batch training is
needed.  This is suitable for many of the classfication model in sklearn that
do not support batch training.  Here I disabled loading of tod data. If it is
not suitable for a particular pipeline that you are working on, please enable
it by

pipe.set_load_data(True)

If you are working on more advanced models that support batch processing,
please refer to the batch_pipe.py.  """

from mlpipe import MLPipe
from models.deep import CNNModel

pipe = MLPipe()

# setup some pipeline parameters
pipe.set_epochs(30)
pipe.set_train_bias(good=2, bad=3)
pipe.set_train_batch_size(1024)
pipe.set_validate_batch_size(1024)
pipe.set_validate_interval(1)

# load dataset
pipe.load_dataset('data/dataset_2d.h5', load_data=True)

# add models to train and test together
pipe.add_model(CNNModel())

# excute the pipeline
pipe.train()

# pipe.test()
pipe.save()
pipe.clean()
