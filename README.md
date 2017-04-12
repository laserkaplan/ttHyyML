# ttHyyML

Machine Learning exercises for ttHyy

## Getting setup

### First time setup on lxplus

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

### Normal setup on lxplus

After setting up the environment for the first time, you can return to this setup by doing `source setup_lxplus.sh`, which is equivalent to:

```
setupATLAS
lsetup root
source ve/bin/activate

export PATH="`pwd`:${PATH}"
export PYTHONPATH="`pwd`:${PYTHONPATH}"
```

## Usage

### Training the neural network

The main steering macro is `train.py`, which will:

1. load the data;
2. split it into testing and training samples;
3. train and test the neural net; and
4. plot and save the ROC curve.

First, you will want to put the input ROOT files into directories called `inputs_leptonic` and `inputs_hadronic`.
It is suggested to use symbolics link to the public directories that the ntuples are located rather than copying them all to the working directory.

`train.py` has a few options:

- `-c, --channel`, which allows you to change which channel you want to train on (leptonic or hadronic);
- `--cat, --categorical`, which allows you to train a categorical model instead of a simple binary selector (not currently supported);
- `-s, --signal`, which allows you to restrict the number of signal event to use relative to the number of background events, to prevent overtraining on the signal;
- `-n, --name`, which changes the name of the saved ROC curve; and
- `--save`, which will save the weights of the neural network to an HDF5 file in the `models` directory.

### Applying the neural network weights

To apply the trained weights to an ntuple, use the macro `applyWeight.py`.
It has the following options:

- `-c, --channel`, which is the same as above for `train.py`;
- `-m, --model`, the path to the input weights file;
- `-i, --input`, the name of the input file (with the `inputs` directory and `.root` stripped!);
- `-t, --tree`, the name of the input tree; and
- `-n, --name`, the name appended to the output file.

Outputs are saved in the `output` folder.

### Models

The models used in the analysis can be found in `ttHyy/models.py`.
Currently, only a shallow model is used.
Deeper models will be added as the need for additional neural network complexity arises.
