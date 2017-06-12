from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

def model_shallow(ndim, doDropout=False):
    model = Sequential()
    model.add(Dense(100, input_dim = ndim))
    model.add(Activation('relu'))
    if doDropout: model.add(Dropout(0.01))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def model_shallow_categorical(ndim, nclasses, doDropout=False):
    model = Sequential()
    model.add(Dense(16, input_dim = ndim))
    model.add(Activation('relu'))
    if doDropout: model.add(Dropout(0.2))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))
    return model

def model_deep_categorical(ndim, nclasses, doDropout=False):
    model = Sequential()
    model.add(Dense(16, input_dim = ndim))
    model.add(Activation('relu'))
    if doDropout: model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    if doDropout: model.add(Dropout(0.2))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))
    return model
