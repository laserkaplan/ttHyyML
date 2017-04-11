import os

import ROOT

import numpy as np

def restrictSample(sample, nbkg, signal):
    '''
    Restrict the size of a list of samples based on the number of background events and a multiplier
    '''
    if signal == 0 or len(sample) < signal * nbkg:
        pass
    else:
        sample = np.random.choice(sample, size=int(signal * nbkg))

    return sample
