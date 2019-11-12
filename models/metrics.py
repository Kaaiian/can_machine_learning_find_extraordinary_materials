import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, \
                            precision_recall_fscore_support, average_precision_score
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


class Meter():
    def __init__(self, props, gaps, model_types):
        self.props = props
        self.gaps = gaps
        self.model_types = model_types
        self.meter = {}
        for prop in self.props:
            self.meter[prop] = {}
            for gap in self.gaps:
                self.meter[prop][gap] = {}
                for model_type in self.model_types:
                    self.meter[prop][gap][model_type] = []
#            if prop == 'ael_bulk_modulus_vrh':
#                self.meter[prop][gap][model_type] = []
#                prop, gap, model_type

    def update(self, prop, gap, model_type, output):
        self.meter[prop][gap][model_type] = output
        print(prop, gap, model_type)

    def metrics(self):
        precision = {}
        recall = {}
        fscore = {}
        roc_auc = {}
        pr_auc = {}
        roc_points = {}
        pr_points = {}
        for prop in self.props:
            precision[prop] = {}
            recall[prop] = {}
            fscore[prop] = {}
            roc_auc[prop] = {}
            pr_auc[prop] = {}
            roc_points[prop] = {}
            pr_points[prop] = {}
            for gap in self.gaps:
                precision[prop][gap] = {}
                recall[prop][gap] = {}
                fscore[prop][gap] = {}
                roc_auc[prop][gap] = {}
                pr_auc[prop][gap] = {}
                roc_points[prop][gap] = {}
                pr_points[prop][gap] = {}
                for model_type in self.model_types:
                    if model_type == 'ridge_density' and prop != 'ael_bulk_modulus_vrh':
                        continue
                    output = self.meter[prop][gap][model_type]

                    # get roc_auc metrics
                    fpr, tpr, _ = roc_curve(output[2], output[3])
                    precisions, recalls, _ = precision_recall_curve(output[2],
                                                                    output[3])
                    roc_points[prop][gap][model_type] = [fpr, tpr]
                    pr_points[prop][gap][model_type] = [recalls, precisions]
                    y_pred_labeled = [1 if x >= output[0] else
                                      0 for x in output[3]]
                    prfs = precision_recall_fscore_support(output[2],
                                                           y_pred_labeled)
                    precision[prop][gap][model_type] = prfs[0]
                    recall[prop][gap][model_type] = prfs[1]
                    fscore[prop][gap][model_type] = prfs[2]
                    roc_auc[prop][gap][model_type] = auc(fpr, tpr)
                    pr_auc[prop][gap][model_type] = average_precision_score(
                                                          output[2], output[3])

        self.precision = precision
        self.recall = recall
        self.fscore = fscore
        self.roc_auc = roc_auc
        self.pr_auc = pr_auc
        self.roc_points = roc_points
        self.pr_points = pr_points

    def plot_curve(self, curve='roc', folder='figures'):
        for prop in self.props:
            for gap in self.gaps:
                plt.figure(figsize=(7, 7))
                for model_type in self.model_types:
                    if model_type == 'ridge_density' and prop != 'ael_bulk_modulus_vrh':
                        continue
                    if curve == 'roc':
                        x, y = self.roc_points[prop][gap][model_type]
                        xlabel = 'false positive rate'
                        ylabel = 'true positive rate'
                        metric = self.roc_auc[prop][gap]
                    elif curve == 'pr':
                        x, y = self.pr_points[prop][gap][model_type]
                        xlabel = 'recall'
                        ylabel = 'precision'
                        metric = self.pr_auc[prop][gap]
                    else:
                        print('wronge curve type')
                        break
                    plt.plot(x, y, label=model_type)
                save_dir =  folder+'/' + prop + '/' + str(gap) + '/'
                os.makedirs(save_dir, exist_ok=True)
                fig_name = save_dir + curve + '_curve.png'
                plt.legend()
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.tick_params(direction='in', top=True, right=True)
                plt.savefig(fig_name, dpi=300)
                print('Area under curve:', metric)

    def save(self, folder='figures'):
        for prop in self.props:
            for gap in self.gaps:
                save_dir = folder+'/' + prop + '/' + str(gap) + '/'
                os.makedirs(save_dir, exist_ok=True)
                columns = ['precision',
                           'recall',
                           'fscore',
                           'roc_auc',
                           'pr_auc']
                df_metrics = pd.DataFrame(columns=columns)
                for model_type in self.model_types:
                    if model_type == 'ridge_density' and prop != 'ael_bulk_modulus_vrh':
                        continue
                    precision = self.precision[prop][gap][model_type]
                    recall = self.recall[prop][gap][model_type]
                    fscore = self.fscore[prop][gap][model_type]
                    roc_auc = self.roc_auc[prop][gap][model_type]
                    pr_auc = self.pr_auc[prop][gap][model_type]
                    df_metrics.loc[model_type, 'precision'] = precision[1]
                    df_metrics.loc[model_type, 'recall'] = recall[1]
                    df_metrics.loc[model_type, 'fscore'] = fscore[1]
                    df_metrics.loc[model_type, 'roc_auc'] = roc_auc
                    df_metrics.loc[model_type, 'pr_auc'] = pr_auc
                df_metrics.to_csv(save_dir+'metrics.csv')
        print('saved metrics to csv')
