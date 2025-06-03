import pandas as pd
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import csv
import os
import warnings
from argparse import ArgumentParser

warnings.filterwarnings("ignore")
global acc


def make_print_to_file(path='./'):
    """
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    """
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

    fileName = datetime.datetime.now().strftime('%Y_%m_%d' + '_Time' + '%H_%M_%S' + '_CM_evaluate')
    sys.stdout = Logger(fileName + '.log', path=path)

    print(fileName.center(60, '*'))


def cal_metrics(confusion_matrix):
    # n = confusion_matrix.shape[0]
    metrics_result = []

    ALL = np.sum(confusion_matrix)

    TP = confusion_matrix[1, 1]

    # FP = np.sum(confusion_matrix[:, i]) - TP
    FP = confusion_matrix[0, 1]

    # FN = np.sum(confusion_matrix[i, :]) - TP
    FN = confusion_matrix[1, 0]

    # TN = ALL - TP - FP - FN
    TN = confusion_matrix[0, 0]
    metrics_result.append(
        ['{:.4f}'.format(TP / (TP + FP)), '{:.4f}'.format(TP / (TP + FN)), '{:.4f}'.format(TN / (TN + FP)),
         '{:.4f}'.format((TP + TN) / ALL)])
    return metrics_result


def main(args):
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    make_print_to_file(path=args.savepath + '/')


    CSV_file = args.csvpath

    print(CSV_file)

    # Fill in the fields according to your own columns
    y_label = pd.read_csv(CSV_file, usecols=['usecols']).to_numpy()
    y_pre = pd.read_csv(CSV_file, usecols=['usecols']).to_numpy()

    thresh = 0.5
    print('thresh:', thresh)

    y_label = np.array([[1 if i > thresh else 0 for i in j] for j in y_label])
    y_pre = np.array([[1 if i > thresh else 0 for i in j] for j in y_pre])

    category_index = ['conjunctival injection', 'pterygium', 'keratitis', 'pingueculae', 'cataract', 'corneal degeneration', 'conjunctival tumor', 'eyelid tumor']
    accuracy_score = metrics.accuracy_score(y_label, y_pre)
    print("accuracy_score:", '{:.4f}'.format(accuracy_score))
    f1_score = metrics.f1_score(y_label, y_pre, average='weighted')
    print("f1_score:", '{:.4f}'.format(f1_score))


    print("classification_report:",
          metrics.classification_report(y_label, y_pre, digits=4, target_names=category_index))
    for index, value in enumerate(category_index):
        kappa_score = metrics.cohen_kappa_score(y_label[:, index], y_pre[:, index])
        print("{:>30}".format(value), "kappa_score:", '{:.4f}'.format(kappa_score))

    cm = multilabel_confusion_matrix(y_label, y_pre)

    print("cm:")
    print(cm)
    for index, value in enumerate(category_index):
        cm = metrics.confusion_matrix(y_label[:, index], y_pre[:, index])

        result = cal_metrics(cm)

        print("{:>30}".format(category_index[index]), ":precision,sensitivity,specificity,acc:", result)


if __name__ == "__main__":
    CSV_PATH = './result.csv'
    SAVE_PATH = './'

    parser = ArgumentParser()
    parser.add_argument('--csvpath', help='csv file', default=CSV_PATH)
    parser.add_argument('--savepath', help='save file', default=SAVE_PATH)
    get_params = parser.parse_args()
    main(get_params)
