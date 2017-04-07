from keras.models import Sequential
from keras.layers import Dense, Activation

def model_leptonic_shallow(ndim, nclasses):
    model = Sequential()
    model.add(Dense(64, input_dim = ndim))
    model.add(Activation('relu'))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))
    return model
