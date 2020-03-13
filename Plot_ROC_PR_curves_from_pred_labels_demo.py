import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.metrics import *
import pandas as pd

now = datetime.datetime.now()
month = str(now.strftime("%b"))
day = str(now.strftime("%d"))
year = str(now.strftime("%y"))


def plot_ROC_PR_curves(preds, labels):
    # Set font style and define figure, axes
    mpl.rc('font', family='Arial')
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # make a list of tuples with pred value and label

    try:
        ll = len(preds[0])

        all_cv_tprs = []
        all_cv_fprs = []
        all_cv_precisions = []
        all_cv_recalls = []

        for jj, preds_i in enumerate(preds):
            labels_i = labels[jj]

            all_predictions = []
            for i in range(len(preds_i)):
                all_predictions.append((preds_i[i], labels_i[i]))

            n_tot_pos = list(labels_i).count(1)
            n_tot_neg = list(labels_i).count(0)

            tp = 0.
            fp = 0.
            fn = 0.
            tn = 0.

            # define lists that will hold values

            all_tprs = []
            all_fprs = []
            all_precisions = []
            all_recalls = []

            # Loop through all predicted probabilities, sorted in descending order
            for i, p_t_hold in enumerate(sorted(all_predictions, reverse=True)):

                # true pos of false pos based on label

                if p_t_hold[1] == 1:
                    tp += 1
                else:
                    fp += 1

                tn = n_tot_neg - fp
                fn = n_tot_pos - tp

                # tpr, fpr, precision, recall defined at every point based on tp, fp, precision and recall

                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)

                precision = tp / (tp + fp)
                recall = tpr

                # add the values to the lists
                all_tprs.append(tpr)
                all_fprs.append(fpr)

                all_precisions.append(precision)
                all_recalls.append(recall)

            # set the initial values to 0

            all_tprs[0] = 0
            all_recalls[0] = 0

            all_cv_fprs.append(all_fprs)
            all_cv_tprs.append(all_tprs)
            all_cv_precisions.append(all_precisions)
            all_cv_recalls.append(all_recalls)

        mean_fpr = np.mean(all_cv_fprs, axis=0)
        mean_tpr = np.mean(all_cv_tprs, axis=0)

        tprs_upper = np.minimum(mean_tpr + np.std(all_cv_tprs, axis=0), 1)
        tprs_lower = np.maximum(mean_tpr - np.std(all_cv_tprs, axis=0), 0)

        all_roc_aucs = [auc(all_cv_fprs[k], all_cv_tprs[k]) for k in range(len(all_cv_tprs))]

        # draw roc curve

        roc_line, = ax[0].plot(mean_fpr, mean_tpr, color='k', alpha=0.8,
                               linewidth=2,
                               linestyle='-', label='AUC = {m} \u00B1 {s}'.format(
                m=round(auc(mean_fpr, mean_tpr), 3),
                s=round(np.std(all_roc_aucs), 2))
                               )

        roc_shading = ax[0].fill_between(mean_fpr, tprs_upper, tprs_lower, alpha=0.1,
                                         color='k', linewidth=0)

        # draw pr curve

        mean_precision = np.mean(all_cv_precisions, axis=0)
        mean_recall = np.mean(all_cv_recalls, axis=0)

        all_pr_aucs = [auc(all_cv_recalls[k], all_cv_precisions[k]) for k in range(len(all_cv_precisions))]

        precisions_upper = np.minimum(mean_precision + np.std(all_cv_precisions, axis=0), 1)
        precisions_lower = np.maximum(mean_precision - np.std(all_cv_precisions, axis=0), 0)

        pr_line, = ax[1].plot(mean_recall, mean_precision, color='k', alpha=0.8,
                              linewidth=2,
                              linestyle='-', label='AUC = {m} \u00B1 {s}'.format(
                m=round(auc(mean_recall, mean_precision), 3),
                s=round(np.std(all_pr_aucs), 2))
                              )

        pr_shading = ax[1].fill_between(mean_recall, precisions_upper, precisions_lower, alpha=0.1,
                                        color='k', linewidth=0)

        ax[0].set_aspect('equal')
        ax[0].set_xlim([-0.02, 1.02])
        ax[0].set_ylim([-0.02, 1.02])
        ax[0].set_xticks(np.linspace(0, 1, 6))
        ax[0].set_xlabel('False Positive Rate', fontsize=20)
        ax[0].set_ylabel('True Positive Rate', fontsize=20)
        ax[0].set_title('ROC Curve', fontsize=20)

        # draw chance line in ROC curve

        c, = ax[0].plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='#bdbdbd', alpha=0.9,
                        label='chance (AUC = 0.5)')

        ax[0].grid(b=True, which='major', color='#bdbdbd', linestyle='--', lw=0.5, alpha=0.3)
        leg1 = ax[0].legend(fontsize=14, loc='lower right')

        ax[1].set_aspect('equal')
        ax[1].set_xlim([-0.02, 1.02])
        ax[1].set_ylim([-0.02, 1.02])
        ax[1].set_xticks(np.linspace(0, 1, 6))
        ax[1].set_xlabel('Recall', fontsize=20)
        ax[1].set_ylabel('Precision', fontsize=20)
        ax[1].set_title('PR Curve', fontsize=20)

        # draw chance line in PR curve

        c, = ax[1].plot([0, 1], [mean_precision[-1], mean_precision[-1]],
                        linestyle='--', lw=1.5, color='#bdbdbd', alpha=0.9,
                        label='chance (AUC = {})'.format(
                            round(auc([0, 1], [mean_precision[-1], mean_precision[-1]]), 3)))

        ax[1].grid(b=True, which='major', color='#bdbdbd', linestyle='--', lw=0.5, alpha=0.3)
        leg2 = ax[1].legend(fontsize=14, loc='center left')

        fig.suptitle('Performance of digenic classifier on various non-digenic sets',
                     fontsize=24)

    except:
        all_predictions = []
        for i in range(len(preds)):
            all_predictions.append((preds[i], labels[i]))

        n_tot_pos = list(labels).count(1)
        n_tot_neg = list(labels).count(0)

        tp = 0.
        fp = 0.
        fn = 0.
        tn = 0.

        # define lists that will hold values

        all_tprs = []
        all_fprs = []
        all_precisions = []
        all_recalls = []

        # Loop through all predicted probabilities, sorted in descending order
        for i, p_t_hold in enumerate(sorted(all_predictions, reverse=True)):

            # true pos of false pos based on label

            if p_t_hold[1] == 1:
                tp += 1
            else:
                fp += 1

            tn = n_tot_neg - fp
            fn = n_tot_pos - tp

            # tpr, fpr, precision, recall defined at every point based on tp, fp, precision and recall

            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            precision = tp / (tp + fp)
            recall = tpr

            # add the values to the lists
            all_tprs.append(tpr)
            all_fprs.append(fpr)

            all_precisions.append(precision)
            all_recalls.append(recall)

        # set the initial values to 0

        all_tprs[0] = 0
        all_recalls[0] = 0

        # draw roc curve

        roc_line, = ax[0].plot(all_fprs, all_tprs, color='k', alpha=0.8,
                               linewidth=2,
                               linestyle='-', label='AUC = {}'.format(round(auc(all_fprs, all_tprs), 3))
                               )

        # draw pr curve

        pr_line, = ax[1].plot(all_recalls, all_precisions, color='k', alpha=0.8,
                              linewidth=2,
                              linestyle='-', label='AUC = {}'.format(round(auc(all_recalls, all_precisions), 3))
                              )

        ax[0].set_aspect('equal')
        ax[0].set_xlim([-0.02, 1.02])
        ax[0].set_ylim([-0.02, 1.02])
        ax[0].set_xticks(np.linspace(0, 1, 6))
        ax[0].set_xlabel('False Positive Rate', fontsize=20)
        ax[0].set_ylabel('True Positive Rate', fontsize=20)
        ax[0].set_title('ROC Curve', fontsize=20)

        # draw chance line in ROC curve

        c, = ax[0].plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='#bdbdbd', alpha=0.9,
                        label='chance (AUC = 0.5)')

        ax[0].grid(b=True, which='major', color='#bdbdbd', linestyle='--', lw=0.5, alpha=0.3)
        leg1 = ax[0].legend(fontsize=14, loc='lower right')

        ax[1].set_aspect('equal')
        ax[1].set_xlim([-0.02, 1.02])
        ax[1].set_ylim([-0.02, 1.02])
        ax[1].set_xticks(np.linspace(0, 1, 6))
        ax[1].set_xlabel('Recall', fontsize=20)
        ax[1].set_ylabel('Precision', fontsize=20)
        ax[1].set_title('PR Curve', fontsize=20)

        # draw chance line in PR curve

        c, = ax[1].plot([0, 1], [all_precisions[-1], all_precisions[-1]],
                        linestyle='--', lw=1.5, color='#bdbdbd', alpha=0.9,
                        label='chance (AUC = {})'.format(
                            round(auc([0, 1], [all_precisions[-1], all_precisions[-1]]), 3)))

        ax[1].grid(b=True, which='major', color='#bdbdbd', linestyle='--', lw=0.5, alpha=0.3)
        leg2 = ax[1].legend(fontsize=14, loc='left')

        fig.suptitle('Performance of digenic classifier on various non-digenic sets',
                     fontsize=24)

    return fig
