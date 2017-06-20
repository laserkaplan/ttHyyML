import os

def signal_multiplier(s):
    s = float(s)
    if s < 0.0:
        raise argparse.ArgumentTypeError('%r must be >= 0!' % s)
    return s

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--channel', action='store', choices=['l', 'lep', 'leptonic', 'h', 'had', 'hadronic'], default='h', help='Channel to process')
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

from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from tabulate import tabulate
    
def train_leptonic():

    #branches = ['N_jet30_cen', 'HT_jet30/1000000', 'mt_lep_met/1000000', 'pt_lep_met/1000000', 'ph_cos_eta2_1', '(ph_pt1+ph_pt2)/1000000','pTt_yy/1000000']
    branches = ['N_jet30_cen', 'HT_jet30/1000000', 'mt_lep_met/1000000', 'pt_lep_met/1000000', '(ph_pt1+ph_pt2)/1000000.']

    # load training data
    print('Loading training data.')

    train_selectionMC   = 'N_lep > 0 && N_bjet30_fixed70 >  0 && N_jet30_cen >= 0 &&  (flag_passedIso && flag_passedPID) && random_number > 1'
    train_selectiondata = 'N_lep > 0 && N_bjet30_fixed70 == 0 && N_jet30_cen >  0 && !(flag_passedIso && flag_passedPID) && random_number > 1'

    train_sig = root2array('all_inputs/ttH.root' , treename='output;5', branches=branches, selection=train_selectionMC  )
    train_bkg = root2array('all_inputs/data.root', treename='output;1', branches=branches, selection=train_selectiondata)
    train_sig = rec2array(train_sig)
    train_bkg = rec2array(train_bkg)

    #load testing data
    test_selectionMC   = 'N_lep > 0 && N_bjet30_fixed70 >  0 && N_jet30_cen >= 0 &&  (flag_passedIso && flag_passedPID) && random_number < 1'
    test_selectiondata = 'N_lep > 0 && N_bjet30_fixed70 == 0 && N_jet30_cen >  0 && !(flag_passedIso && flag_passedPID) && random_number < 1'

    test_sig = root2array('all_inputs/ttH.root' , treename='output;5', branches=branches, selection=test_selectionMC  )
    test_bkg = root2array('all_inputs/data.root', treename='output;1', branches=branches, selection=test_selectiondata)
    test_sig = rec2array(test_sig)
    test_bkg = rec2array(test_bkg)

    # split data into train, val, and test samples
    headers = ['Sample', 'Total', 'Training', 'Testing']
    sample_size_table = [
        ['Signal'    , len(train_sig)+len(test_sig), len(train_sig), len(test_sig)],
        ['Background', len(train_bkg)+len(test_bkg), len(train_bkg), len(test_bkg)],
    ]
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')

    # organize data for training
    print('Organizing data for training.')

    train = np.concatenate((train_sig, train_bkg))
    test  = np.concatenate((test_sig , test_bkg ))

    y_train_cat = np.concatenate((np.zeros(len(train_sig), dtype=np.uint8), np.ones(len(train_bkg), dtype=np.uint8)))
    y_test_cat  = np.concatenate((np.zeros(len(test_sig) , dtype=np.uint8), np.ones(len(test_bkg) , dtype=np.uint8)))

    # train model
    print('Train model.')
    model = models.model_shallow(len(branches), True)
    rms = optimizers.RMSprop(lr=0.0001)
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train, y_train_cat, epochs=1000, batch_size=32, validation_data=(test, y_test_cat), callbacks=[early_stopping])
    model.summary()

    # test model
    print('Test model.')
    score = model.predict(test, batch_size=1000)

    # plot ROC curve
    print('Plotting ROC curve.')
    pltname = 'ROC_curve_leptonic_' + args.name
    utils.plotROC(y_test_cat, score, pltname, False)

    #plot loss function
    print('Plotting loss function')
    pltname = 'Loss_function_leptonic_' + args.name
    utils.plotLoss(history.history['loss'], history.history['val_loss'], pltname, False)

    #plot accuracy function
    print('Plotting accuracy function')
    pltname = 'Accuracy_function_leptonic_' + args.name
    utils.plotAcc(history.history['acc'], history.history['val_acc'], pltname, False)

    # save model
    if args.save:
        print('Saving model')
        mdlname = 'models/model_leptonic_shallow_'+args.name+'.h5'
        model.save_weights(mdlname)

    return

def train_hadronic():

    #input branches
    branches = ['N_jet30', 'N_jet30_cen', 'N_bjet30_fixed70', 'HT_jet30/1000000.', 'mass_jet30/1000000.', 'ph_delt_eta2_1/3.']
    #branches = ['N_jet30', 'N_jet30_cen', 'N_bjet30_fixed70', 'HT_jet30/1000000', 'mass_jet30/1000000', 'pTt_yy/1000000', 'ph_cos_eta2_1','(ph_pt1+ph_pt2)/1000000']

    # load training data
    print('Loading training data.')

    train_selectionMC   = 'N_lep == 0 && N_jet30 >= 3 && N_bjet30_fixed70 > 0 &&  (flag_passedIso && flag_passedPID) && random_number > 1'
    train_selectiondata = 'N_lep == 0 && N_jet30 >= 3 && N_bjet30_fixed70 > 0 && !(flag_passedIso && flag_passedPID) && random_number > 1'

    train_bkg_data = root2array('all_inputs/data.root' , treename='output;1' , branches=branches, selection=train_selectiondata)
    train_bkg_ggH  = root2array('all_inputs/ggH.root'  , treename='output;4' , branches=branches, selection=train_selectionMC  )
    train_sig      = root2array('all_inputs/ttH.root'  , treename='output;5' , branches=branches, selection=train_selectionMC  )
    train_bkg      = np.concatenate((train_bkg_data, train_bkg_ggH))

    train_sig = rec2array(train_sig)
    train_bkg = rec2array(train_bkg)

    # load testing data
    print('Loading testing data.')

    test_selectionMC   = 'N_lep == 0 && N_jet30 >= 3 && N_bjet30_fixed70 > 0 &&  (flag_passedIso && flag_passedPID) && random_number < 1'
    test_selectiondata = 'N_lep == 0 && N_jet30 >= 3 && N_bjet30_fixed70 > 0 && !(flag_passedIso && flag_passedPID) && random_number < 1'

    test_bkg_data = root2array('all_inputs/data.root' , treename='output;1' , branches=branches, selection=test_selectiondata)
    test_bkg_ggH  = root2array('all_inputs/ggH.root'  , treename='output;4' , branches=branches, selection=test_selectionMC  )
    test_sig      = root2array('all_inputs/ttH.root'  , treename='output;5' , branches=branches, selection=test_selectionMC  )
    test_bkg      = np.concatenate((test_bkg_data, test_bkg_ggH))

    test_sig = rec2array(test_sig)
    test_bkg = rec2array(test_bkg)

    #table for easy number readout
    headers = ['Sample', 'Total', 'Training', 'Testing']
    sample_size_table = [
        ['Signal'    , len(train_sig)+len(test_sig), len(train_sig), len(test_sig)],
        ['Background', len(train_bkg)+len(test_bkg), len(train_bkg), len(test_bkg)],
    ]
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')

    # organize data for training
    print('Organizing data for training.')

    train = np.concatenate((train_sig, train_bkg))
    test  = np.concatenate((test_sig , test_bkg ))

    y_train_cat = np.concatenate((np.zeros(len(train_sig), dtype=np.uint8), np.ones(len(train_bkg), dtype=np.uint8)))
    y_test_cat  = np.concatenate((np.zeros(len(test_sig) , dtype=np.uint8), np.ones(len(test_bkg) , dtype=np.uint8)))

    # train model
    print('Train model.')

    model = models.model_shallow(len(branches), True)
    rms = optimizers.RMSprop(lr=0.0001)
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train, y_train_cat, epochs=1000, batch_size=1000, validation_data=(test, y_test_cat), callbacks=[early_stopping])

    # test model
    print('Test model.')
    score = model.predict(test, batch_size=1000)

    # plot ROC curve
    print('Plotting ROC curve.')
    pltname = 'ROC_curve_hadronic_' + args.name
    utils.plotROC(y_test_cat, score, pltname, False)

    #plot loss function
    print('Plotting loss function')
    pltname = 'Loss_function_hadronic_' + args.name
    utils.plotLoss(history.history['loss'], history.history['val_loss'], pltname, False)

    #plot accuracy function
    print('Plotting accuracy function')
    pltname = 'Accuracy_function_hadronic_' + args.name
    utils.plotAcc(history.history['acc'], history.history['val_acc'], pltname, False)

    # save model
    if args.save:
        print('Saving model')
        mdlname = 'models/model_hadronic_shallow_'+args.name+'.h5'
        model.save_weights(mdlname)

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
