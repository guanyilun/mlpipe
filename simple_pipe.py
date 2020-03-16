"""
This script is the boilerplate for a simple pipeline where no batch training is
needed.  This is suitable for many of the classfication model in sklearn that
do not support batch training.  Here I disabled loading of tod data. If it is
not suitable for a particular pipeline that you are working on, please enable
it by

pipe.set_load_data(True)

If you are working on more advanced models that support batch processing,
please refer to the batch_pipe.py.  """

# set matplotlib style
import matplotlib
import matplotlib.style
matplotlib.style.use("classic")

from mlpipe import MLPipe
from mlpipe.models.randfor import RFModel
from mlpipe.models.xgb import XGBModel
# from mlpipe.models.tree import DecisionTreeModel
# from mlpipe.models.knn import KNNModel

pipe = MLPipe()

# since there is no batch training
# epotches and batch size are not
# important
pipe.set_epochs(1)
pipe.set_train_batch_size(0)
pipe.set_train_bias(good=1, bad=1)

# in this pipeline I will not need
# the tod data
pipe.load_dataset('./data/merged.h5',
                  load_data=False)

# add models to train and test together
pipe.add_model(XGBModel(name="XGBoost_tuned",
                        scale_pos_weight=1,
                        learning_rate=0.2,
                        colsample_bytree=0.4,
                        subsample=0.8,
                        objective='binary:logistic',
                        n_estimators=5000,
                        reg_alpha=0.3,
                        max_depth=4,
                        gamma=10))
pipe.add_model(XGBModel())
pipe.add_model(RFModel(name="RF-5",
                       n_estimators=5))
pipe.add_model(RFModel(name="RF-7",
                       n_estimators=7))
pipe.add_model(RFModel(name="RF-12",
                       n_estimators=12))

# excute the pipeline
pipe.train()

# only test after fine tuning
pipe.test()
# pipe.save()
pipe.clean()
