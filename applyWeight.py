import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', action='store', required=True, help='HDF5 file with model weights to apply')
parser.add_argument('-i', '--input', action='store', required=True, help='Name of input ROOT file to apply weights to')
parser.add_argument('-t', '--tree', action='store', default='output', help='Name of tree inside input file')
parser.add_argument('-n', '--name', action='store', required=True, help='Name to be appended to output file') 
args = parser.parse_args()

import ROOT

from array import array

import numpy as np

from ttHyy import models

def main():
    # load model
    model = models.model_shallow(4, True)
    model.load_weights(args.model)

    # open file and get tree
    infile = ROOT.TFile.Open('inputs/%s.root' % args.input)
    intree = infile.Get(args.tree)

    # create new tree
    outtree = ROOT.TTree(args.name, args.name)
    outtree.SetDirectory(0)

    # create branches for tree
    m_yy          = array('f', [0])
    N_j_central30 = array('i', [0])
    N_j_btag30    = array('i', [0])
    ph_isTight1   = array('i', [0])
    ph_iso1       = array('i', [0])
    ph_isTight2   = array('i', [0])
    ph_iso2       = array('i', [0])
    weight        = array('f', [0])
    nnweight      = array('f', [0])
    outtree.Branch('m_yy'         , m_yy         , 'm_yy/F'         )
    outtree.Branch('N_j_central30', N_j_central30, 'N_j_central30/I')
    outtree.Branch('N_j_btag30'   , N_j_btag30   , 'N_j_btag30/I'   )
    outtree.Branch('ph_isTight1'  , ph_isTight1  , 'ph_isTight1/I'  )
    outtree.Branch('ph_iso1'      , ph_iso1      , 'ph_iso1/I'      )
    outtree.Branch('ph_isTight2'  , ph_isTight2  , 'ph_isTight2/I'  )
    outtree.Branch('ph_iso2'      , ph_iso2      , 'ph_iso2/I'      )
    outtree.Branch('weight'       , weight       , 'weight/F'       )
    outtree.Branch('nnweight'     , nnweight     , 'nnweight/F'     )

    # loop over input tree and fill new tree
    for event in intree:
        test = np.array([[event.N_j_central30, event.m_HT_30 / 1000.0, event.m_mT / 1000.0, event.m_pTlepEtmiss / 1000.0]])
        score = model.predict(test)
        m_yy[0]          = event.m_yy
        N_j_central30[0] = event.N_j_central30
        N_j_btag30[0]    = event.N_j_btag30
        ph_isTight1[0]   = event.ph_isTight1
        ph_iso1[0]       = event.ph_iso1
        ph_isTight2[0]   = event.ph_isTight2
        ph_iso2[0]       = event.ph_iso2
        weight[0]        = event.m_weight
        nnweight[0]      = score[0]
        outtree.Fill()

    # save new tree
    outfile = ROOT.TFile('outputs/%s_%s.root' % (args.input, args.name), 'RECREATE')
    outtree.Write()
    outfile.Close()

    return

if __name__ == '__main__':
    main()
