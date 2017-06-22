from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Masking, LSTM, concatenate

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
    m_in = Input(shape=(ndim,))
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

def model_rnn(indata):
    m_in = Input(shape=indata[0].shape)
    m = Masking(mask_value=-999)(m_in)
    m = LSTM(32)(m)
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=m_in, outputs=m_out)
    return model

def model_rnn_with_photons(indata_jets, indata_photons):
    m_in_jets = Input(shape=indata_jets[0].shape, name='in_jets')
    m_jets = Masking(mask_value=-999)(m_in_jets)
    m_jets = LSTM(32)(m_jets)
    m_jets = Dense(64)(m_jets)
    m_jets = Activation('relu')(m_jets)
    m_jets = Dropout(0.2)(m_jets)

    #m_in_photons = Input(shape=indata_photons[0].shape, name='in_photons')
    m_in_photons = Input(shape=(1,), name='in_photons')

    m = concatenate([m_jets, m_in_photons])
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=[m_in_jets, m_in_photons], outputs=m_out)
    return model
