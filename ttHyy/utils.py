import os

import ROOT

import numpy as np

from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt

import itertools

def restrictSample(sample, nbkg, signal):
    '''
    Restrict the size of a list of samples based on the number of background events and a multiplier
    '''
    if signal == 0 or len(sample) < signal * nbkg:
        pass
    else:
        sample = np.random.choice(sample, size=int(signal * nbkg))

    return sample

def plotROC(y_test, score, filename, show=False):
    fpr, tpr, _ = roc_curve(y_test, score)
    roc_auc = auc(fpr, tpr)

    fpr = 1.0 - fpr

    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.plot(tpr, fpr, label='Shallow NN, area = %0.2f' % roc_auc)
    plt.plot([0, 1], [1, 0], linestyle='--', color='black', label='Luck')
    plt.xlabel('Signal acceptance')
    plt.ylabel('Background rejection')
    plt.title('Receiver operating characteristic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.legend(loc='lower left', framealpha=1.0)

    plt.savefig('plots/' + filename + '.png')
    plt.savefig('plots/' + filename + '.eps')
    
    if show: plt.show()

    return

def plotCM(y_test, score, classes, filename, show=False):
    cm = confusion_matrix(y_test, score)
    diag = float(np.trace(cm)) / float(np.sum(cm))
    cm = cm.T.astype('float') / cm.T.sum(axis=0)
    cm = np.rot90(cm.T, 1)
    
    np.set_printoptions(precision=2)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix, diagonal = %.2f' % (100 * diag))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes[::-1])
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.2f' % (100 * cm[i, j]),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > 100 * thresh else 'black')

    plt.xlabel('True event label')
    plt.ylabel('Event classification')
    plt.tight_layout()

    plt.savefig('plots/' + filename + '.png')
    plt.savefig('plots/' + filename + '.eps')

    if show: plt.show()

    return
