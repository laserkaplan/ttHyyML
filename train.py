import os

def signal_multiplier(s):
    s = float(s)
    if s < 0.0:
        raise argparse.ArgumentTypeError('%r must be >= 0!' % s)
    return s

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--channel', action='store', choices=['l', 'lep', 'leptonic', 'h', 'had', 'hadronic'], default='l', help='Channel to process')
parser.add_argument('--cat', '--categorical', action='store_true', help='Create categorical model')
parser.add_argument('-s', '--signal', action='store', type=signal_multiplier, default='5', help='Number of signal events to use in training as a multiple of the number of background events. Value of 0 means use all signal events.')
args = parser.parse_args()

import ROOT
from root_numpy import root2array, rec2array

import numpy as np

from ttHyy.models import model_shallow

from sklearn import model_selection
from sklearn.metrics import roc_curve, auc

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
    
def train_leptonic():
    # load data
    print('Loading data.')

    branches = ['N_j_central30', 'm_HT_30/1000', 'm_mT/1000', 'm_pTlepEtmiss/1000']
    selection = 'm_nlep >= 1 && N_j_central30 >= 1 && N_j_btag30 == 0'
    selectiondata = selection + ' && (ph_isTight1 != 0 || ph_iso1 != 0 || ph_isTight2 != 0 || ph_iso2 != 0)'

    sig = root2array('inputs/ttHrw.root'       , treename='output;5', branches=branches, selection=selection    )
    bkg = root2array('inputs/data_looserw.root', treename='output'  , branches=branches, selection=selectiondata)

    print('Number of background events = %d' % len(bkg))
    if args.signal == 0 or len(sig) < args.signal * len(bkg): 
        print('Number of signal events = %d' % len(sig))
    else: 
        print('Restricting number of signal events to %r * %d = %d' % (args.signal, len(bkg), int(args.signal * len(bkg))))
        sig = np.random.choice(sig, size=int(args.signal * len(bkg)))
    
    sig = rec2array(sig)
    bkg = rec2array(bkg)

    # split data into train, val, and test samples
    print('Splitting data.')

    train_sig, test_sig = model_selection.train_test_split(sig, test_size = 0.3, random_state=1234)
    train_bkg, test_bkg = model_selection.train_test_split(bkg, test_size = 0.3, random_state=1234)
    val_sig, test_sig = np.split(test_sig, [len(test_sig) / 2])
    val_bkg, test_bkg = np.split(test_bkg, [len(test_bkg) / 2])

    # organize data for training
    print('Organizing data for training.')

    train = np.concatenate((train_sig, train_bkg))
    val   = np.concatenate((val_sig  , val_bkg  ))
    test  = np.concatenate((test_sig , test_bkg ))

    '''
    y_train = np.concatenate((np.zeros(len(train_sig), dtype=np.uint8), np.ones(len(train_bkg), dtype=np.uint8)))
    y_val   = np.concatenate((np.zeros(len(val_sig)  , dtype=np.uint8), np.ones(len(val_bkg)  , dtype=np.uint8)))
    y_test  = np.concatenate((np.zeros(len(test_sig) , dtype=np.uint8), np.ones(len(test_bkg) , dtype=np.uint8)))

    y_train_cat = to_categorical(y_train, 2)
    y_val_cat   = to_categorical(y_val  , 2)
    y_test_cat  = to_categorical(y_test , 2)
    '''
    
    y_train_cat = np.concatenate((np.zeros(len(train_sig), dtype=np.uint8), np.ones(len(train_bkg), dtype=np.uint8)))
    y_val_cat   = np.concatenate((np.zeros(len(val_sig)  , dtype=np.uint8), np.ones(len(val_bkg)  , dtype=np.uint8)))
    y_test_cat  = np.concatenate((np.zeros(len(test_sig) , dtype=np.uint8), np.ones(len(test_bkg) , dtype=np.uint8)))

    # train model
    print('Train model.')

    model = model_shallow(4, True)
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train, y_train_cat, epochs=20, batch_size=32, validation_data=(val, y_val_cat))

    # test model
    print('Test model.')

    score = model.predict(test, batch_size=32)

    # plot ROC curve
    print('Plotting ROC curve')

    fpr, tpr, _ = roc_curve(y_test_cat, score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='Shallow NN, area = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Luck')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.show()

    return

def main():
    if args.channel[0] == 'l':
        if args.cat: 
            print('No support for categorical model yet!')
            return
        else:
            print('Training binary leptonic model.')
            train_leptonic()
    else:
        print('No support for hadronic model yet!')
        return

    return

if __name__ == '__main__':
    main()
