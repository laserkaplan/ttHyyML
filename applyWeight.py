import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--channel', action='store', choices=['l', 'lep', 'leptonic', 'h', 'had', 'hadronic'], default='h', help='Channel to process')
parser.add_argument('-m', '--model', action='store', required=True, help='HDF5 file with model weights to apply')
parser.add_argument('-i', '--input', action='store', required=True, help='Name of input ROOT file to apply weights to')
parser.add_argument('-t', '--tree', action='store', default='output', help='Name of tree inside input file')
parser.add_argument('-n', '--name', action='store', required=True, help='Name to be appended to output file') 
args = parser.parse_args()

import math

import ROOT
from root_numpy import root2array, rec2array

from array import array

import numpy as np

from ttHyy import models

def process_leptonic():
    # load model
    model = models.model_shallow(5, True)
    model.load_weights(args.model)

    # open file and get tree
    infile = ROOT.TFile.Open('all_inputs/%s.root' % args.input)
    intree = infile.Get(args.tree)

    # create new tree
    outtree = ROOT.TTree(args.name, args.name)
    outtree.SetDirectory(0)

    # create branches for tree
    m_yy          = array('f', [0])
    N_j_central30 = array('i', [0])
    N_j_btag30    = array('i', [0])
    N_lep         = array('i', [0])
    ph_isTight1   = array('i', [0])
    ph_iso1       = array('i', [0])
    ph_isTight2   = array('i', [0])
    ph_iso2       = array('i', [0])
    weight        = array('f', [0])
    nnweight      = array('f', [0])
    random_number = array('f', [0])
    outtree.Branch('m_yy'         , m_yy         , 'm_yy/F'         )
    outtree.Branch('N_j_central30', N_j_central30, 'N_j_central30/I')
    outtree.Branch('N_j_btag30'   , N_j_btag30   , 'N_j_btag30/I'   )
    outtree.Branch('N_lep'        , N_lep        , 'N_lep/I'        )
    outtree.Branch('ph_isTight1'  , ph_isTight1  , 'ph_isTight1/I'  )
    outtree.Branch('ph_iso1'      , ph_iso1      , 'ph_iso1/I'      )
    outtree.Branch('ph_isTight2'  , ph_isTight2  , 'ph_isTight2/I'  )
    outtree.Branch('ph_iso2'      , ph_iso2      , 'ph_iso2/I'      )
    outtree.Branch('weight'       , weight       , 'weight/F'       )
    outtree.Branch('nnweight'     , nnweight     , 'nnweight/F'     )
    outtree.Branch('random_number', random_number, 'random_number/F')

    #setup scores
    branches = [
        'N_jet30_cen',
        'HT_jet30/1000000',
        'mt_lep_met/1000000',
        'pt_lep_met/1000000',
        'ph_delt_eta2_1/3',
        'pTt_yy/1000000',
        '(ph_pt1+ph_pt2)/1000000',
        'mass_yy',
        'N_bjet30_fixed70',
        'N_lep',
        'flag_passedIso',
        'flag_passedPID',
        'weight',
        'random_number'
    ]
    inX = rec2array(root2array('all_inputs/%s.root' % args.input, treename=args.tree, branches=branches))
    score = model.predict(inX[:, :5])
    inX = np.column_stack((inX, score))

    # loop over input tree and fill new tree
    for event in inX:
        m_yy[0]          = event[7]
        N_j_central30[0] = int(event[0])
        N_j_btag30[0]    = int(event[8])
        N_lep[0]         = int(event[9])
        ph_isTight1[0]   = int(event[11])
        ph_iso1[0]       = int(event[10])
        ph_isTight2[0]   = int(event[11])
        ph_iso2[0]       = int(event[10])
        weight[0]        = event[12]
        random_number[0] = event[13]
        nnweight[0]      = event[14]
        outtree.Fill()

    # save new tree
    outfile = ROOT.TFile('outputs/%s_leptonic_%s.root' % (args.input, args.name), 'RECREATE')
    outtree.Write()
    outfile.Close()
    print 'Making outputs/%s_leptonic_%s.root' % (args.input, args.name)
    return

def process_hadronic():
    # load model
    model = models.model_shallow(6, True)
    model.load_weights(args.model)

    # open file and get tree
    infile = ROOT.TFile.Open('all_inputs/%s.root' % args.input)
    intree = infile.Get(args.tree)

    # create new tree
    outtree = ROOT.TTree(args.name, args.name)
    outtree.SetDirectory(0)

    # create branches for tree
    m_yy          = array('f', [0])
    N_lep         = array('i', [0])
    N_j_30        = array('i', [0])
    N_j_btag30    = array('i', [0])
    ph_isTight1   = array('i', [0])
    ph_iso1       = array('i', [0])
    ph_isTight2   = array('i', [0])
    ph_iso2       = array('i', [0])
    weight        = array('f', [0])
    random_number = array('f', [0])
    nnweight      = array('f', [0])
    outtree.Branch('m_yy'         , m_yy         , 'm_yy/F'         )
    outtree.Branch('N_lep'        , N_lep        , 'N_lep/I'        )
    outtree.Branch('N_j_30'       , N_j_30       , 'N_j_30/I'       )
    outtree.Branch('N_j_btag30'   , N_j_btag30   , 'N_j_btag30/I'   )
    outtree.Branch('ph_isTight1'  , ph_isTight1  , 'ph_isTight1/I'  )
    outtree.Branch('ph_iso1'      , ph_iso1      , 'ph_iso1/I'      )
    outtree.Branch('ph_isTight2'  , ph_isTight2  , 'ph_isTight2/I'  )
    outtree.Branch('ph_iso2'      , ph_iso2      , 'ph_iso2/I'      )
    outtree.Branch('weight'       , weight       , 'weight/F'       )
    outtree.Branch('random_number', random_number, 'random_number/F')
    outtree.Branch('nnweight'     , nnweight     , 'nnweight/F'     )

    # get NN score
    branches = [
        'N_jet30',
        'N_jet30_cen',
        'N_bjet30_fixed70',
        'HT_jet30/1000000',
        'mass_jet30/1000000',
        'ph_delt_eta2_1/3.',
        'pTt_yy',
        '(ph_pt1+ph_pt2)/1000000',
        'mass_yy',
        'N_lep',
        'flag_passedIso',
        'flag_passedPID',
        'weight',
        'random_number'
    ]
    inX = rec2array(root2array('all_inputs/%s.root' % args.input, treename=args.tree, branches=branches))
    score = model.predict(inX[:, :6])
    inX = np.column_stack((inX, score))

    # loop over input tree and fill new tree
    for event in inX:
        m_yy[0]         = event[8]
        N_lep[0]        = int(event[9])
        N_j_30[0]       = int(event[0])
        N_j_btag30[0]   = int(event[2])
        ph_isTight1[0]  = int(event[11])
        ph_iso1[0]      = int(event[10])
        ph_isTight2[0]  = int(event[11])
        ph_iso2[0]      = int(event[10])
        weight[0]       = event[12]
        random_number[0]= event[13] 
        nnweight[0]     = event[14]
        outtree.Fill()

    # save new tree
    outfile = ROOT.TFile('outputs/%s_hadronic_%s.root' % (args.input, args.name), 'RECREATE')
    outtree.Write()
    outfile.Close()
    print 'Making outputs/%s_hadronic_%s.root' % (args.input, args.name)

    return

def main():
    if args.channel[0] == 'l':
        process_leptonic()
    else:
        process_hadronic()

    return

if __name__ == '__main__':
    main()
