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

import pickle

from sklearn import model_selection

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tabulate import tabulate

def train_hadronic():
    # load data
    print('Loading data.')

    '''
    branches = []
    for i in range(8): branches.append('jet_%d' % (i+1))
    selectionMC   = 'N_lep == 0 && N_jet30 > 3'
    selectiondata = 'N_lep == 0 && N_jet30 > 3 && (flag_passedIso == 0 || flag_passedPID == 0)'

    sig = root2array('inputs/ttH_RNN.root' , branches=branches, selection=selectionMC  )
    bkg = root2array('inputs/data_RNN.root', branches=branches, selection=selectiondata)

    sig_jets = process_jets(sig, branches)
    '''

    sig = np.load('arrays/ttH_hadronic_jets.npy')
    bkg_data = np.load('arrays/data_IsoPID_hadronic_jets.npy')
    bkg_ggH = np.load('arrays/ggH_hadronic_jets.npy')

    bkg = np.concatenate((bkg_data, bkg_ggH))
    
    #construct records
    sig = utils.restrictSample(sig, len(bkg), args.signal)

    # scale branches
    from sklearn.preprocessing import StandardScaler

    pts = np.concatenate((sig[:, :, 0].flatten(), bkg[:, :, 0].flatten()))
    mask_pts = pts != -999
    scaler_pt = StandardScaler()
    scaler_pt.fit(pts[mask_pts])
    
    Es = np.concatenate((sig[:, :, 0].flatten(), bkg[:, :, 0].flatten()))
    mask_Es = Es != -999
    scaler_E = StandardScaler()
    scaler_E.fit(Es[mask_Es])

    scalers = {'pt': scaler_pt, 'E': scaler_E}
    pickle.dump(scalers, open('scalers.p', 'wb'))

    pts_sig = sig[:, :, 0].flatten()
    mask_sig = pts_sig != -999
    pts_sig[mask_sig] = scaler_pt.transform(pts_sig[mask_sig])
    sig[:, :, 0] = pts_sig.reshape(-1, 9)
    
    Es_sig = sig[:, :, 3].flatten()
    Es_sig[mask_sig] = scaler_E.transform(Es_sig[mask_sig])
    sig[:, :, 3] = Es_sig.reshape(-1, 9)
    
    pts_bkg = bkg[:, :, 0].flatten()
    mask_bkg = pts_bkg != -999
    pts_bkg[mask_bkg] = scaler_pt.transform(pts_bkg[mask_bkg])
    bkg[:, :, 0] = pts_bkg.reshape(-1, 9)
    
    Es_bkg = bkg[:, :, 3].flatten()
    Es_bkg[mask_bkg] = scaler_E.transform(Es_bkg[mask_bkg])
    bkg[:, :, 3] = Es_bkg.reshape(-1, 9)

    '''
    sig = rec2array(sig)
    bkg = rec2array(bkg)
    '''

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

    model = models.model_rnn(train)
    model.summary()
    rmsprop = RMSprop(lr=0.0005)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=20)
    mc = ModelCheckpoint('models/model_hadronic_rnn_temp.h5', monitor='val_loss', save_best_only=True)
    try:
        model.fit(train, y_train_cat, epochs=1000, batch_size=128, validation_data=(val, y_val_cat), callbacks=[es, mc])
    except KeyboardInterrupt:
        print('Finishing on SIGINT.')

    # test model
    print('Test model.')

    score = model.predict(test, batch_size=128)

    # plot ROC curve
    print('Plotting ROC curve.')
    pltname = 'ROC_curve_hadronic_' + args.name
    utils.plotROC(y_test_cat, score, pltname, False)

    # save model
    if args.save:
        print('Saving model')
        model.save_weights('models/model_hadronic_rnn.h5')

    return

'''
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
'''

def main():
    train_hadronic()
    
    return

if __name__ == '__main__':
    main()
