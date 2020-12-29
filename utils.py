import importlib
import os
from string import Template

from tabulate import tabulate

import matplotlib.pyplot as plt

from config import TRAIN_TEMP_DATA_BASE_PATH


def load_dotted_path(path):
    split_path = path.split(".")
    modulename, classname = ".".join(split_path[:-1]), split_path[-1]
    model = getattr(importlib.import_module(modulename), classname)
    return model


def training_header():
    """Prints a training header to the console"""
    print('| Epoch | Avg. loss | Train Acc. | Test Acc.  | Elapsed  |   ETA    |\n'
          '|-------+-----------+------------+------------+----------+----------|')


def training_epoch_stats(epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
    """This function is called every time an epoch ends, all of the important training statistics are taken as
    an input and outputted to the console.
   :param epoch: current training epoch epoch
   :type epoch: int
   :param running_loss: current total running loss of the model
   :type running_loss: float
   :param total: number of the images that model has trained on
   :type total: int
   :param train_acc: models accuracy on the provided train dataset
   :type train_acc: str
   :param test_acc: models accuracy on the optionally provided eval dataset
   :type test_acc: str
   :param elapsed: total time it took to run the epoch
   :type elapsed: str
   :param eta: an estimated time when training will end
   :type eta: str
   """
    print('| {:5d} | {:.7f} | {:10s} | {:10s} | {:8s} | {} |'
          .format(epoch, running_loss / total, train_acc, test_acc, elapsed, eta))


def training_finished(start, now):
    """Outputs elapsed training time to the console.
    :param start: time when the training has started
    :type start: datetime.datetime
    :param now: current time
    :type now: datetime.datetime
    """
    print('Training finished, total time elapsed: {}\n'.format(now - start))


def accuracy_summary_basic(total, correct, acc):
    """Outputs model evaluation statistics to the console if verbosity is set to 1.
    :param total: number of total objects model was evaluated on
    :type total: int
    :param correct: number of correctly classified objects
    :type correct: int
    :param acc: overall accuracy of the model
    :type acc: float
    """
    print('Total: {} -- Correct: {} -- Accuracy: {}%\n'.format(total, correct, acc))


def accuracy_summary_extended(classes, class_total, class_correct):
    """Outputs model evaluation statistics to the console if  verbosity is equal or greater than 2.
    :param classes: list with class names on which model was evaluated
    :type classes: list
    :param class_total: list with a number of classes on which model was evaluated
    :type class_total: list
    :param class_correct: list with a number of properly classified classes
    :type class_correct: list
    """
    print('| {} | {} | {} |'.format(type(classes), type(class_total), type(class_correct)))
    table = []
    for i, c in enumerate(classes):
        if class_total[i] != 0:
            class_acc = '{:.1f}%'.format(100 * class_correct[i] / class_total[i])
        else:
            class_acc = '-'
        table.append([c, class_total[i], class_correct[i], class_acc])
    print(tabulate(table, headers=['Class', 'Total', 'Correct', 'Acc'], tablefmt='orgtbl'), '\n')


def strfdelta(tdelta, fmt='%H:%M:%S'):
    """Similar to strftime, but this one is for a datetime.timedelta object.
    :param tdelta: datetime object containing some time difference
    :type tdelta: datetime.timedelta
    :param fmt: string with a format
    :type fmt: str
    :return: string containing a time in a given format
    """

    class DeltaTemplate(Template):
        delimiter = "%"

    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def save_train_loss_graph(train_loss, filename):
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(os.path.join(TRAIN_TEMP_DATA_BASE_PATH, f'{filename}_loss.png'))
