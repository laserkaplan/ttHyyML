from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout

def model_shallow(ndim):
    m_in = Input(shape=(ndim,))
    m = Dense(32)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_shallow_categorical(ndim, nclasses):
    m_in = Input(shape(ndim,))
    m = Dense(32)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(nclasses)(m)
    m_out = Activation('softmax')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_deep_categorical(ndim, nclasses):
    m_in = Input(shape=(ndim,))
    m = Dense(32)(m_in)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(nclasses)(m)
    m_out = Activation('softmax')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model
