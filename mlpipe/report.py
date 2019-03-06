from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn import metrics
from tabulate import tabulate
import os
from matplotlib import pyplot as plt
import scikitplot as skplt
from .utils import to_categorical
import cPickle as pickle


class Report(object):
    def __init__(self, output_dir="outputs"):
            
        columns = ['epoch', 'batch', 'model', 'loss', 'base', 'accuracy',
                   'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1', 'time/s']

        self.report = pd.DataFrame(columns=columns)
        self.output_dir = output_dir

    def add_record(self, model_name, epoch, batch, predict, proba,
                   truth, time_spent, plot=True, roc_ax=None, pr_ax=None):
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
        if plot and np.any(proba):
            if not os.path.exists(self.output_dir):
                print("Folder %s doesn't exist, creating now..." % self.output_dir)
                os.makedirs(self.output_dir)

            # plot roc curve
            skplt.metrics.plot_roc(truth, proba)
            filename = os.path.join(self.output_dir, "%s_roc_curve.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)

            # plot precision-recall curve
            skplt.metrics.plot_precision_recall(truth, proba)
            filename = os.path.join(self.output_dir, "%s_pr_curve.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)

            # plot confusion matrix
            skplt.metrics.plot_confusion_matrix(y_true=truth, y_pred=predict)
            filename = os.path.join(self.output_dir, "%s_confusion_matrix.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)

            # plot cumulative gain curve
            skplt.metrics.plot_cumulative_gain(truth, proba)
            filename = os.path.join(self.output_dir, "%s_cumulative_gain.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)

            # plot lift curve
            skplt.metrics.plot_lift_curve(truth, proba)
            filename = os.path.join(self.output_dir, "%s_lift_curve.png" % model_name)
            print("Saving plot: %s" % filename)
            plt.savefig(filename)

            # if external ax is given, it means that we want to plot cross
            # model performance comparison
            if roc_ax:
                # plot a cross model ROC curve, with only the micro average
                # not the individual classes
                truth_binarize = to_categorical(truth, 2)
                fpr, tpr, _ = metrics.roc_curve(truth_binarize.ravel(), proba.ravel())
                roc_auc = metrics.auc(fpr, tpr)
                roc_ax.plot(fpr, tpr, label='{0} (area = {1:0.2f})'.format(model_name, roc_auc),
                            linestyle='-', linewidth=2)

                roc_param = {
                    'name': model_name,
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc
                }
                # save the parameter used for external use
                filename = os.path.join(self.output_dir, "%s_roc.pickle"%model_name)
                print("Saving data: %s" % filename)
                with open(filename, "wb") as f:
                    pickle.dump(roc_param, f)

            if pr_ax:
                # plot a cross model PR curve, with only the micro average
                # not the individual classes
                truth_binarize = to_categorical(truth, 2)
                precision, recall, _ = metrics.precision_recall_curve(
                    truth_binarize.ravel(), proba.ravel())
                average_precision = metrics.average_precision_score(truth_binarize,
                                                                    proba,
                                                                    average='micro')
                pr_ax.plot(recall, precision,
                           label='{0} (area = {1:0.3f})'.format(model_name, average_precision),
                           linestyle='-', linewidth=2)

                pr_param = {
                    'name': model_name,
                    'recall': recall,
                    'precision': precision,
                    'auc': average_precision
                }
                # save the parameter used for external use
                filename = os.path.join(self.output_dir, "%s_pr.pickle"%model_name)
                print("Saving data: %s" % filename)
                with open(filename, "wb") as f:
                    pickle.dump(pr_param, f)

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
