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
parser.add_argument('-n', '--name', action='store', default='test', help='Name of output plot.')
parser.add_argument('--save', action='store_true', help='Save model weights to HDF5 file')
args = parser.parse_args()

import ROOT
from root_numpy import root2array, rec2array

import numpy as np

from ttHyy import *

from sklearn import model_selection

from keras.utils.np_utils import to_categorical

from tabulate import tabulate
    
def train_leptonic():
    # load data
    print('Loading data.')

    branches = ['N_j_central30', 'm_HT_30/1000', 'm_mT/1000', 'm_pTlepEtmiss/1000']
    selectionMC   = 'N_lep > 0 && N_j_btag30 > 0'
    selectiondata = 'N_lep > 0 && N_j_btag30 == 0 && N_j_central30 > 0 && (ph_isTight1 == 0 || ph_iso1 == 0 || ph_isTight2 == 0 || ph_iso2 == 0)'

    sig = root2array('inputs_leptonic/ttHrw.root'       , treename='output;5', branches=branches, selection=selectionMC  )
    bkg = root2array('inputs_leptonic/data_looserw.root', treename='output'  , branches=branches, selection=selectiondata)

    sig = utils.restrictSample(sig, len(bkg), args.signal)

    sig = rec2array(sig)
    bkg = rec2array(bkg)

    # split data into train, val, and test samples
    print('Splitting data.')

    train_sig, test_sig = model_selection.train_test_split(sig, test_size = 0.3, random_state=1234)
    train_bkg, test_bkg = model_selection.train_test_split(bkg, test_size = 0.3, random_state=1234)
    val_sig, test_sig = np.split(test_sig, [len(test_sig) / 2])
    val_bkg, test_bkg = np.split(test_bkg, [len(test_bkg) / 2])

    headers = ['Sample', 'Total', 'Training', 'Validation', 'Testing']
    sample_size_table = [
        ['Signal'    , len(sig), len(train_sig), len(val_sig), len(test_sig)],
        ['Background', len(bkg), len(train_bkg), len(val_bkg), len(test_bkg)],
    ]
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')

    # organize data for training
    print('Organizing data for training.')

    train = np.concatenate((train_sig, train_bkg))
    val   = np.concatenate((val_sig  , val_bkg  ))
    test  = np.concatenate((test_sig , test_bkg ))

    y_train_cat = np.concatenate((np.zeros(len(train_sig), dtype=np.uint8), np.ones(len(train_bkg), dtype=np.uint8)))
    y_val_cat   = np.concatenate((np.zeros(len(val_sig)  , dtype=np.uint8), np.ones(len(val_bkg)  , dtype=np.uint8)))
    y_test_cat  = np.concatenate((np.zeros(len(test_sig) , dtype=np.uint8), np.ones(len(test_bkg) , dtype=np.uint8)))

    # train model
    print('Train model.')

    model = models.model_shallow(4, True)
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train, y_train_cat, epochs=20, batch_size=32, validation_data=(val, y_val_cat))

    # test model
    print('Test model.')

    score = model.predict(test, batch_size=32)

    # plot ROC curve
    print('Plotting ROC curve.')

    pltname = 'ROC_curve_leptonic_' + args.name
    utils.plotROC(y_test_cat, score, pltname, True)

    # save model
    if args.save:
        print('Saving model')
        model.save_weights('models/model_leptonic_shallow.h5')

    return

def train_leptonic_categorical():
    # load data
    print('Loading data.')

    branches = ['N_j_central30', 'm_HT_30/1000', 'm_mT/1000', 'm_pTlepEtmiss/1000']
    selectionMC   = 'N_lep > 0 && N_j_btag30 > 0'
    selectiondata = 'N_lep > 0 && N_j_btag30 == 0 && N_j_central30 > 0 && (ph_isTight1 == 0 || ph_iso1 == 0 || ph_isTight2 == 0 || ph_iso2 == 0)'

    ttH  = root2array('inputs_leptonic/ttHrw.root'        , treename='output;5', branches=branches, selection=selectionMC  )
    tHjb = root2array('inputs_leptonic/tHjbrw.root'       , treename='output'  , branches=branches, selection=selectionMC  )
    tWH  = root2array('inputs_leptonic/tWHrw.root'        , treename='output'  , branches=branches, selection=selectionMC  )
    bkg  = root2array('inputs_leptonic/data_looserw.root' , treename='output'  , branches=branches, selection=selectiondata)

    tH = np.concatenate((tHjb, tWH))

    leastevents = min(len(ttH), len(tH), len(bkg))

    ttH = utils.restrictSample(ttH, leastevents, args.signal)
    tH  = utils.restrictSample(tH , leastevents, args.signal)
    bkg = utils.restrictSample(bkg, leastevents, args.signal)

    ttH = rec2array(ttH)
    tH  = rec2array(tH )
    bkg = rec2array(bkg)

    # split data into train, val, and test samples
    print('Splitting data.')

    train_ttH, test_ttH = model_selection.train_test_split(ttH, test_size = 0.3, random_state=1234)
    train_tH , test_tH  = model_selection.train_test_split(tH , test_size = 0.3, random_state=1234)
    train_bkg, test_bkg = model_selection.train_test_split(bkg, test_size = 0.3, random_state=1234)
    val_ttH, test_ttH = np.split(test_ttH, [len(test_ttH) / 2])
    val_tH , test_tH  = np.split(test_tH , [len(test_tH ) / 2])
    val_bkg, test_bkg = np.split(test_bkg, [len(test_bkg) / 2])

    headers = ['Sample', 'Total', 'Training', 'Validation', 'Testing']
    sample_size_table = [
        ['ttH'       , len(ttH), len(train_ttH), len(val_ttH), len(test_ttH)],
        ['tH'        , len(tH ), len(train_tH ), len(val_tH ), len(test_tH )],
        ['Background', len(bkg), len(train_bkg), len(val_bkg), len(test_bkg)],
    ]
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')

    # organize data for training
    print('Organizing data for training.')

    train = np.concatenate((train_ttH, train_tH, train_bkg))
    val   = np.concatenate((val_ttH  , val_tH  , val_bkg  ))
    test  = np.concatenate((test_ttH , test_tH , test_bkg ))

    y_train = np.concatenate((
        np.zeros(len(train_ttH), dtype=np.uint8),
        np.ones(len(train_tH), dtype=np.uint8),
        np.ones(len(train_bkg), dtype=np.uint8) + 1,
    ))
    y_val = np.concatenate((
        np.zeros(len(val_ttH), dtype=np.uint8),
        np.ones(len(val_tH), dtype=np.uint8),
        np.ones(len(val_bkg), dtype=np.uint8) + 1,
    ))
    y_test = np.concatenate((
        np.zeros(len(test_ttH), dtype=np.uint8),
        np.ones(len(test_tH), dtype=np.uint8),
        np.ones(len(test_bkg), dtype=np.uint8) + 1,
    ))

    y_train_cat = to_categorical(y_train, 3)
    y_val_cat   = to_categorical(y_val  , 3)
    y_test_cat  = to_categorical(y_test , 3)
    
    # train model
    print('Train model.')

    model = models.model_shallow_categorical(4, 3, True)
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train, y_train_cat, epochs=20, batch_size=32, validation_data=(val, y_val_cat))

    # test model
    print('Test model.')

    score = model.predict_classes(test, batch_size=32)

    # plot confusion matrix
    print('Plotting confusion matrix.')

    pltname = 'CM_leptonic_' + args.name
    classes = ['ttH', 'tH', 'Background']
    utils.plotCM(y_test, score, classes, pltname, True)

    # save model
    if args.save:
        print('Saving model')
        model.save_weights('models/model_leptonic_shallow_categorical.h5')

    return

def train_hadronic():
    # load data
    print('Loading data.')

    branches = ['N_j_30', 'N_j_central30', 'N_j_btag30', 'm_HT_30/1000000', 'm_alljet/1000000', 'm_met/sqrt(m_HT_30)']
    selectionMC   = 'N_lep == 0 && N_j_30 >= 3 && N_j_btag30 > 0'
    selectiondata = 'N_lep == 0 && N_j_30 >= 3 && N_j_btag30 > 0 && (ph_isTight1 == 0 || ph_iso1 == 0 || ph_isTight2 == 0 || ph_iso2 == 0)'

    sig      = root2array('inputs_hadronic/ttH.root'             , treename='output;14', branches=branches, selection=selectionMC  )
    bkg_data = root2array('inputs_hadronic/data_LooseLepton.root', treename='output'   , branches=branches, selection=selectiondata)
    bkg_ggH  = root2array('inputs_hadronic/ggH.root'             , treename='output;8' , branches=branches, selection=selectionMC  )

    bkg = np.concatenate((bkg_data, bkg_ggH))

    sig = utils.restrictSample(sig, len(bkg), args.signal)

    sig = rec2array(sig)
    bkg = rec2array(bkg)

    # split data into train, val, and test samples
    print('Splitting data.')

    train_sig, test_sig = model_selection.train_test_split(sig, test_size = 0.3, random_state=1234)
    train_bkg, test_bkg = model_selection.train_test_split(bkg, test_size = 0.3, random_state=1234)
    val_sig, test_sig = np.split(test_sig, [len(test_sig) / 2])
    val_bkg, test_bkg = np.split(test_bkg, [len(test_bkg) / 2])

    headers = ['Sample', 'Total', 'Training', 'Validation', 'Testing']
    sample_size_table = [
        ['Signal'    , len(sig), len(train_sig), len(val_sig), len(test_sig)],
        ['Background', len(bkg), len(train_bkg), len(val_bkg), len(test_bkg)],
    ]
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')

    # organize data for training
    print('Organizing data for training.')

    train = np.concatenate((train_sig, train_bkg))
    val   = np.concatenate((val_sig  , val_bkg  ))
    test  = np.concatenate((test_sig , test_bkg ))

    y_train_cat = np.concatenate((np.zeros(len(train_sig), dtype=np.uint8), np.ones(len(train_bkg), dtype=np.uint8)))
    y_val_cat   = np.concatenate((np.zeros(len(val_sig)  , dtype=np.uint8), np.ones(len(val_bkg)  , dtype=np.uint8)))
    y_test_cat  = np.concatenate((np.zeros(len(test_sig) , dtype=np.uint8), np.ones(len(test_bkg) , dtype=np.uint8)))

    # train model
    print('Train model.')

    model = models.model_shallow(6, True)
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train, y_train_cat, epochs=100, batch_size=32, validation_data=(val, y_val_cat))

    # test model
    print('Test model.')

    score = model.predict(test, batch_size=32)

    # plot ROC curve
    print('Plotting ROC curve.')

    pltname = 'ROC_curve_hadronic_' + args.name
    utils.plotROC(y_test_cat, score, pltname, True)

    # save model
    if args.save:
        print('Saving model')
        model.save_weights('models/model_hadronic_shallow.h5')

    return

def main():
    if args.channel[0] == 'l':
        if args.cat: 
            print('Training categorical leptonic model.')
            train_leptonic_categorical()
        else:
            print('Training binary leptonic model.')
            train_leptonic()
    else:
        if args.cat: 
            print('No support for categorical model yet!')
            return
        else:
            print('Training binary hadronic model.')
            train_hadronic()

    return

if __name__ == '__main__':
    main()
