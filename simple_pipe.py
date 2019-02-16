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
from models.randfor import RFModel
from models.tree import DecisionTreeModel
from models.xgb import XGBModel

pipe = MLPipe()

# since there is no batch training
# epotches and batch size are not
# important
pipe.set_epochs(1)
pipe.set_train_batch_size(0)
pipe.set_train_bias(good=1, bad=1)

# in this pipeline I will not need
# the tod data
# pipe.load_dataset('data/dataset.h5', load_data=False)
pipe.load_dataset('/mnt/act3/users/yilun/share/dataset.h5', load_data=False)

# add models to train and test together
pipe.add_model(XGBModel())
pipe.add_model(RFModel(n_estimators=5))
pipe.add_model(DecisionTreeModel())

# excute the pipeline
pipe.train()

# pipe.test()
# pipe.save()
pipe.clean()
