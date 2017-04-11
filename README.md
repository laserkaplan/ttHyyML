# ttHyyML
Machine Learning exercises for ttHyy

## First time setup on lxplus
First checkout the code:

```
git clone https://github.com/laserkaplan/ttHyyML.git
cd ttHyyML
```

Then, setup the virtualenv:

```
setupATLAS
lsetup root
lsetup python
virtualenv --python=python2.7 ve
source ve/bin/activate
```

Then, checkout necessary packages:

```
pip install pip --upgrade
pip install theano keras h5py sklearn matplotlib tabulate
pip install --upgrade https://github.com/rootpy/root_numpy/zipball/master
```

If this is the first time you are using keras, you will want to change the backend to theano instead of the default tensorflow.
To do this, edit the appropriate line in `~/.keras/keras.json`.

## Normal setup on lxplus

After setting up the environment for the first time, you can return to this setup by doing `source setup_lxplus.sh`, which is equivalent to:

```
setupATLAS
lsetup root
source ve/bin/activate

export PATH="`pwd`:${PATH}"
export PYTHONPATH="`pwd`:${PYTHONPATH}"
```

## Usage

The main steering macro is `train.py`, which will:

1. load the data,
2. split it into testing and training samples,
3. train and test the neural net, and
4. plot and save the ROC curve.

First, you will want to put the input ROOT files into a directory called `inputs`.
It is suggested to use a symbolic link to the public directory that the ntuples are located rather than copying them all to the working directory.

`train.py` has a few options:

- `-c, --channel`, which allows you to change which channel you want to train on (leptonic or hadronic).  Currently only leptonic is supported.
- `--cat, --categorical`, which allows you to train a categorical model instead of a simple binary selector.  This is currently not supported.
- `-s, --signal`, which allows you to restrict the number of signal event to use relative to the number of background events, to prevent overtraining on the signal.
- `-n, --name`, which changes the name of the saved ROC curve.

## Models

The models used in the analysis can be found in `ttHyy/models.py`.
Currently, only a shallow model is used.
Deeper models will be added as the need for additional neural network complexity arises.
