from __future__ import print_function
import pandas as pd
from sklearn import metrics
from tabulate import tabulate
import os
from matplotlib import pyplot as plt
import scikitplot as skplt


class Report(object):
    def __init__(self, output_dir="outputs", plot=True):
            
        columns = ['epoch', 'batch', 'model', 'loss', 'base', 'accuracy',
                   'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1', 'time/s']

        self.plot = plot
        self.report = pd.DataFrame(columns=columns)
        self.output_dir = output_dir
            

    def add_record(self, model_name, epoch, batch, predict, proba, truth, time_spent):
        loss = metrics.log_loss(truth, predict)
        accuracy = metrics.accuracy_score(truth, predict)
        precision = metrics.precision_score(truth, predict)
        recall = metrics.recall_score(truth, predict)
        f1 = metrics.f1_score(truth, predict)
        tn, fp, fn, tp = metrics.confusion_matrix(truth, predict).ravel()
        base = sum(truth) * 1.0 / len(truth)
        next_index = len(self.report.index)
        
        self.report.loc[next_index] = [epoch, batch, model_name, loss, base, accuracy,
                                       tp, tn, fp, fn, precision, recall, f1, time_spent]

        if self.plot:
            if not os.path.exists(self.output_dir):
                print("Folder %s doesn't exist, creating now..." % self.output_dir)
                os.makedirs(self.output_dir)

            # plot roc curve
            plt.figure()
            skplt.metrics.plot_roc(truth, proba)
            filename = os.path.join(self.output_dir, "%s_roc_curve.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)
            plt.close()

            # plot precision-recall curve
            plt.figure()
            skplt.metrics.plot_precision_recall_curve(truth, proba)
            filename = os.path.join(self.output_dir, "%s_pr_curve.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)
            plt.close()

            # plot confusion matrix
            plt.figure()
            skplt.metrics.plot_confusion_matrix(y_true=truth, y_pred=predict)
            filename = os.path.join(self.output_dir, "%s_confusion_matrix.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)
            plt.close()

            # plot cumulative gain curve
            plt.figure()
            skplt.metrics.plot_cumulative_gain(truth, proba)
            filename = os.path.join(self.output_dir, "%s_cumulative_gain.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)
            plt.close()


    def print_batch_report(self, epoch, batch):
        report = self.report
        mini_report = report[(report.epoch == epoch) & (report.batch == batch)]
        print('')
        print(tabulate(mini_report, headers='keys', tablefmt='pqsl', showindex=False))
        print('')
    
    def save(self):
        self.report.to_pickle(os.path.join(self.output_dir, "summary.pickle"))
                                   
    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
