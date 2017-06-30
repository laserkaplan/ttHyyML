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

def model_rnn_with_aux(indata_jets, indata_aux_len):
    m_in_jets = Input(shape=indata_jets[0].shape, name='in_jets')
    m_jets = Masking(mask_value=-999)(m_in_jets)
    m_jets = LSTM(32)(m_jets)
    m_jets = Dense(64)(m_jets)
    m_jets = Activation('relu')(m_jets)
    m_jets = Dropout(0.2)(m_jets)

    #m_in_aux = Input(shape=indata_aux[0].shape, name='in_aux')
    m_in_aux = Input(shape=(indata_aux_len,), name='in_aux')

    m = concatenate([m_jets, m_in_aux])
    m = Dense(64)(m)
    m = Activation('relu')(m)
    m = Dropout(0.2)(m)
    m = Dense(1)(m)
    m_out = Activation('sigmoid')(m)

    model = Model(inputs=[m_in_jets, m_in_aux], outputs=m_out)
    return model
