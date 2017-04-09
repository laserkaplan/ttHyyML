import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--channel', action='store', choices=['l', 'lep', 'leptonic', 'h', 'had', 'hadronic'], default='l', help='Channel to process')
parser.add_argument('--cat', '--categorical', action='store_true', help='Create categorical model')
args = parser.parse_args()

import ROOT
import numpy as np
from root_numpy import root2array, rec2array
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from ttHyy.models import model_shallow

def main():
    # load data
    print('Loading data.')

    branches = ['N_j_central30', 'm_HT_30/1000', 'm_mT/1000', 'm_pTlepEtmiss/1000']

    sig = rec2array(root2array('inputs/ttHrw.root' , treename='output;5', branches=branches))
    bkg = rec2array(root2array('inputs/datarw.root', treename='output'  , branches=branches))

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

    model = model_shallow(4, 2, True)
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
