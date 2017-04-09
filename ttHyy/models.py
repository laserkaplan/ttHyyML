from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

def model_shallow(ndim, nclasses, doDropout=False):
    model = Sequential()
    model.add(Dense(16, input_dim = ndim))
    model.add(Activation('relu'))
    if doDropout:
        model.add(Dropout(0.2))
    if nclasses == 2:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
    else: 
        model.add(Dense(nclasses))
        model.add(Activation('softmax'))
    return model
