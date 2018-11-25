from __future__ import print_function
import pandas as pd
from sklearn import metrics
from tabulate import tabulate


class Report(object):
    def __init__(self):
        
        columns = ['epoch', 'batch', 'model', 'loss', 'base', 'accuracy',
                   'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1']
            
        self.report = pd.DataFrame(columns=columns)

    def add_record(self, model_name, epoch, batch, predict, truth):
        loss = metrics.log_loss(truth, predict)
        accuracy = metrics.accuracy_score(truth, predict)
        precision = metrics.precision_score(truth, predict)
        recall = metrics.recall_score(truth, predict)
        f1 = metrics.f1_score(truth, predict)
        tn, fp, fn, tp = metrics.confusion_matrix(truth, predict).ravel()
        base = sum(truth) * 1.0 / len(truth)
        next_index = len(self.report.index)
        
        self.report.loc[next_index] = [epoch, batch, model_name, loss, base, accuracy,
                                       tp, tn, fp, fn, precision, recall, f1]
                                   
    def print_batch_report(self, epoch, batch):
        report = self.report
        mini_report = report[(report.epoch == epoch) & (report.batch == batch)]
        print('')
        print(tabulate(mini_report, headers='keys', tablefmt='pqsl', showindex=False))
        print('')
    
    def save(self, fname):
        self.report.to_pickle(fname)
                                   
                                   
                                   
