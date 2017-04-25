import os

import ROOT

import numpy as np

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

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
