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

import math

import ROOT
from root_numpy import root2array

import numpy as np

from ttHyy import *

import pickle

from sklearn import model_selection

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tabulate import tabulate

def process_photons(photons):
    ptts = []

    for p in photons:
        tlv1 = ROOT.TLorentzVector()
        tlv1.SetPtEtaPhiE(p['photon_1'][0], p['photon_1'][1], p['photon_1'][2], p['photon_1'][3])
        tlv2 = ROOT.TLorentzVector()
        tlv2.SetPtEtaPhiE(p['photon_2'][0], p['photon_2'][1], p['photon_2'][2], p['photon_2'][3])
        ptts.append(math.fabs(tlv1.Px() * tlv2.Py() - tlv1.Py() * tlv2.Px()) / (tlv1 - tlv2).Pt() * 2.0)
        pass

    return np.array(ptts)

def train_hadronic():
    # load jet data
    print('Loading jet data.')

    sig_jets = np.load('arrays/ttH_hadronic_jets.npy')
    bkg_data_jets = np.load('arrays/data_IsoPID_hadronic_jets.npy')
    bkg_ggH_jets = np.load('arrays/ggH_hadronic_jets.npy')

    bkg_jets = np.concatenate((bkg_data_jets, bkg_ggH_jets))
    
    sig_jets = utils.restrictSample(sig_jets, len(bkg_jets), args.signal)

    # scale branches
    from sklearn.preprocessing import StandardScaler

    pts = np.concatenate((sig_jets[:, :, 0].flatten(), bkg_jets[:, :, 0].flatten()))
    mask_pts = pts != -999
    scaler_pt = StandardScaler()
    scaler_pt.fit(pts[mask_pts])
    
    Es = np.concatenate((sig_jets[:, :, 0].flatten(), bkg_jets[:, :, 0].flatten()))
    mask_Es = Es != -999
    scaler_E = StandardScaler()
    scaler_E.fit(Es[mask_Es])

    pts_sig = sig_jets[:, :, 0].flatten()
    mask_sig = pts_sig != -999
    pts_sig[mask_sig] = scaler_pt.transform(pts_sig[mask_sig])
    sig_jets[:, :, 0] = pts_sig.reshape(-1, 9)
    
    Es_sig = sig_jets[:, :, 3].flatten()
    Es_sig[mask_sig] = scaler_E.transform(Es_sig[mask_sig])
    sig_jets[:, :, 3] = Es_sig.reshape(-1, 9)
    
    pts_bkg = bkg_jets[:, :, 0].flatten()
    mask_bkg = pts_bkg != -999
    pts_bkg[mask_bkg] = scaler_pt.transform(pts_bkg[mask_bkg])
    bkg_jets[:, :, 0] = pts_bkg.reshape(-1, 9)
    
    Es_bkg = bkg_jets[:, :, 3].flatten()
    Es_bkg[mask_bkg] = scaler_E.transform(Es_bkg[mask_bkg])
    bkg_jets[:, :, 3] = Es_bkg.reshape(-1, 9)

    # load photon data
    print('Loading photon data.')

    branches = ['photon_1', 'photon_2']
    selectionMC   = 'N_lep == 0 && N_jet30 >= 3 && N_bjet30_fixed70 > 0'
    selectiondata = 'N_lep == 0 && N_jet30 >= 3 && N_bjet30_fixed70 > 0 && (flag_passedIso == 0 || flag_passedPID == 0)'

    sig_photons = root2array('inputs/ttH_RNN.root', branches=branches, selection=selectionMC)
    bkg_data_photons = root2array('inputs/data_RNN.root', branches=branches, selection=selectiondata)
    bkg_ggH_photons = root2array('inputs/ggH_RNN.root', branches=branches, selection=selectionMC)

    bkg_photons = np.concatenate((bkg_data_photons, bkg_ggH_photons))
    
    sig_photons = utils.restrictSample(sig_photons, len(bkg_photons), args.signal)

    # get ptts
    sig_ptts = process_photons(sig_photons)
    bkg_ptts = process_photons(bkg_photons)

    # scale branches

    ptts = np.concatenate((sig_ptts, bkg_ptts))
    scaler_ptt = StandardScaler()
    scaler_ptt.fit(ptts)

    sig_photons = scaler_ptt.transform(sig_ptts)
    bkg_photons = scaler_ptt.transform(bkg_ptts)

    # save scalers for later use
    scalers = {'pt': scaler_pt, 'E': scaler_E, 'ptt': scaler_ptt}
    pickle.dump(scalers, open('scalers.p', 'wb'))

    # split data into train, val, and test samples
    print('Splitting data.')

    train_sig_jets, test_sig_jets = model_selection.train_test_split(sig_jets, test_size = 0.3, random_state=1234)
    train_bkg_jets, test_bkg_jets = model_selection.train_test_split(bkg_jets, test_size = 0.3, random_state=1234)
    val_sig_jets, test_sig_jets = np.split(test_sig_jets, [len(test_sig_jets) / 2])
    val_bkg_jets, test_bkg_jets = np.split(test_bkg_jets, [len(test_bkg_jets) / 2])

    headers = ['Sample', 'Total', 'Training', 'Validation', 'Testing']
    sample_size_table = [
        ['Signal'    , len(sig_jets), len(train_sig_jets), len(val_sig_jets), len(test_sig_jets)],
        ['Background', len(bkg_jets), len(train_bkg_jets), len(val_bkg_jets), len(test_bkg_jets)],
    ]
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')
    
    train_sig_photons, test_sig_photons = model_selection.train_test_split(sig_photons, test_size = 0.3, random_state=1234)
    train_bkg_photons, test_bkg_photons = model_selection.train_test_split(bkg_photons, test_size = 0.3, random_state=1234)
    val_sig_photons, test_sig_photons = np.split(test_sig_photons, [len(test_sig_photons) / 2])
    val_bkg_photons, test_bkg_photons = np.split(test_bkg_photons, [len(test_bkg_photons) / 2])

    # organize data for training
    print('Organizing data for training.')

    train_jets = np.concatenate((train_sig_jets, train_bkg_jets))
    val_jets   = np.concatenate((val_sig_jets  , val_bkg_jets  ))
    test_jets  = np.concatenate((test_sig_jets , test_bkg_jets ))

    y_train_cat = np.concatenate((np.zeros(len(train_sig_jets), dtype=np.uint8), np.ones(len(train_bkg_jets), dtype=np.uint8)))
    y_val_cat   = np.concatenate((np.zeros(len(val_sig_jets)  , dtype=np.uint8), np.ones(len(val_bkg_jets)  , dtype=np.uint8)))
    y_test_cat  = np.concatenate((np.zeros(len(test_sig_jets) , dtype=np.uint8), np.ones(len(test_bkg_jets) , dtype=np.uint8)))

    train_photons = np.concatenate((train_sig_photons, train_bkg_photons))
    val_photons   = np.concatenate((val_sig_photons  , val_bkg_photons  ))
    test_photons  = np.concatenate((test_sig_photons , test_bkg_photons ))

    # train model
    print('Train model.')

    model = models.model_rnn_with_photons(train_jets, train_photons)
    model.summary()
    rmsprop = RMSprop(lr=0.0005)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=20)
    mc = ModelCheckpoint('models/model_hadronic_rnn_with_photons_temp.h5', monitor='val_loss', save_best_only=True)
    try:
        model.fit([train_jets, train_photons], y_train_cat, epochs=1000, batch_size=128, validation_data=([val_jets, val_photons], y_val_cat), callbacks=[es, mc])
    except KeyboardInterrupt:
        print('Finishing on SIGINT.')

    # test model
    print('Test model.')

    score = model.predict([test_jets, test_photons], batch_size=128)

    # plot ROC curve
    print('Plotting ROC curve.')
    pltname = 'ROC_curve_hadronic_' + args.name
    utils.plotROC(y_test_cat, score, pltname, False)

    # save model
    if args.save:
        print('Saving model')
        model.save_weights('models/model_hadronic_rnn_with_photons.h5')

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
