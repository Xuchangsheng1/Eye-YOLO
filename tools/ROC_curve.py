import pandas as pd
from statistics import mean
from numpy import interp
from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib.font_manager import FontProperties
from argparse import ArgumentParser
import os


# surppose chinese font
# mpl.rcParams['font.sans-serif'] = ['SimHei']


def make_print_to_file(path='./'):
    """
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    """
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('%Y_%m_%d' + '_Time' + '%H_%M_%S' + 'ROC_95CI_evaluate')
    sys.stdout = Logger(fileName + '.log', path=path)

    print(fileName.center(60, '*'))


def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = metrics.roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics


def main(args):
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    make_print_to_file(path=args.savepath + '/')
    CSV_file = args.csvpath
    # Fill in the fields according to your own columns
    p_csv = pd.read_csv(CSV_file, usecols=['usecols'])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_label = dict()
    y_pre = dict()

    n_classes = ['0', '1', '2', '3', '4', '5', '6', '7']
    category_index = ['conjunctival injection', 'pterygium', 'keratitis', 'pingueculae', 'cataract', 'corneal degeneration', 'conjunctival tumor', 'eyelid tumor']


    for i, v in enumerate(category_index):
        y_label[i] = p_csv['g' + i].to_numpy()
        y_pre[i] = p_csv['d' + i].to_numpy()
        fpr[i], tpr[i], thersholds = metrics.roc_curve(y_label[i], y_pre[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    y_L = []
    y_P = []
    for item in y_label.values():
        y_L.append(item)
    y_L = np.array(y_L)
    for item in y_pre.values():
        # print(item)
        y_P.append(item)
    y_P = np.array(y_P)
    # print(y_L)
    # print(y_P)
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_L.ravel(), y_P.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'navy', 'black', 'darkgrey', 'red', 'chocolate', 'yellow', 'cyan',
              'forestgreen', 'lightskyblue', 'darkblue', 'pink', 'grey', 'olive', 'purple', 'Maroon', 'FireBrick',
              'IndianRed', 'Salmon', 'SaddleBrown', 'Orange', 'DarkKhaki', 'LawnGreen', 'DarkCyan', 'DeepSkyBlue',
              'DarkTurquoise', 'SteelBlue', 'RoyalBlue']
    lw = 2
    plt.figure()
    plt.tight_layout()

    for i in n_classes:
        statistics = bootstrap_auc(y_label[i], y_pre[i], n_classes)
        Mean = mean(np.mean(statistics, axis=1))
        Max = mean(np.max(statistics, axis=1))
        Min = mean(np.min(statistics, axis=1))
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=1,
                 label='{0} AUC = {1:0.4f} (95% CI:{2:0.4f}-{3:0.4f})'
                       ''.format(category_index[int(i)], roc_auc[i], Min, Max))
        print("{:>30}".format(category_index[int(i)]), '{0:0.4f}({1:0.4f}-{2:0.4f})'
                                                       ''.format(roc_auc[i], Min, Max))

    statistics = bootstrap_auc(y_L.ravel(), y_P.ravel(), n_classes)
    Mean = mean(np.mean(statistics, axis=1))
    Max = mean(np.max(statistics, axis=1))
    Min = mean(np.min(statistics, axis=1))
    plt.plot(fpr["micro"], tpr["micro"],
             label='Average AUC = {1:0.4f} (95% CI:{2:0.4f}-{3:0.4f})'
                   ''.format('III', roc_auc["micro"], Min, Max),
             color='deeppink', linestyle=':', linewidth=4)
    print("{:>30}".format('Average AUC'), '{0:0.4f}({1:0.4f}-{2:0.4f})'
                                          ''.format(roc_auc["micro"], Min, Max))


    # plt.plot([0.5, 1], [0.5, 1], 'k--', lw=lw)
    plt.xlim([-0.005, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    plt.xlabel('1 - Specificity')
    # plt.ylabel('True Positive Rate')
    plt.ylabel('Sensitivity')
    plt.title(args.name)
    # plt.title('ROCs for pterygium grading system in SPB')

    plt.legend(bbox_to_anchor=(1, 0), loc="lower right", fontsize=6)

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    savepath = args.savepath + '/ROC_95CI.png'
    save_svg = args.savepath + '/ROC_95CI.svg'
    plt.savefig(savepath, dpi=300)
    plt.savefig(save_svg, dpi=300)


if __name__ == "__main__":
    CSV_PATH = './result.csv'
    SAVE_PATH = './'
    NAME = 'ROCs for Eye-YOLO'

    parser = ArgumentParser()
    parser.add_argument('--csvpath', help='csv file', default=CSV_PATH)
    parser.add_argument('--savepath', help='save file', default=SAVE_PATH)
    parser.add_argument('--name', help='name', default=NAME)
    get_params = parser.parse_args()
    main(get_params)
