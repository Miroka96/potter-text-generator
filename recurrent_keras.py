from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse

try:
    from lstm.RNN_utils import *
except:
    # for direct execution
    from RNN_utils import *


def parse_arguments():
    # Parsing arguments for Network definition
    ap = argparse.ArgumentParser()
    ap.add_argument('-data_dir', default='../final_data/allBooks.txt')
    ap.add_argument('-batch_size', type=int, default=50)
    ap.add_argument('-layer_num', type=int, default=2)
    ap.add_argument('-seq_length', type=int, default=50)
    ap.add_argument('-hidden_dim', type=int, default=500)
    ap.add_argument('-generate_length', type=int, default=200)
    ap.add_argument('-nb_epoch', type=int, default=20)
    ap.add_argument('-mode', default='train')
    ap.add_argument('-weights', default='')
    ap.add_argument('-epochs_per_iteration', type=int, default=10)
    args = vars(ap.parse_args())
    return args


args = parse_arguments()

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
SEQ_LENGTH = args['seq_length']
HIDDEN_DIM = args['hidden_dim']
GENERATE_LENGTH = args['generate_length']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']
WEIGHTS = args['weights']
EPI = args['epochs_per_iteration']

# Creating training data
X, y, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)
VOCAB_SIZE = len(ix_to_char)


def create_model():
    # Creating and compiling the Network
    model = Sequential()
    model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    for i in range(LAYER_NUM - 1):
        model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    return model


MODEL = create_model()


def initialize_model():
    if not WEIGHTS == '':
        # Loading the trained weights
        MODEL.load_weights(WEIGHTS)
        epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
        print('Loaded weights for epoch {}'.format(epoch))
    else:
        epoch = 0
    return epoch


actual_epoch = initialize_model()

print('sample text (generated):')
# Generate some sample before training to know how bad it is!
generate_text(MODEL, GENERATE_LENGTH, ix_to_char)

if MODE == 'train' or WEIGHTS == '':
    while True:
        print('\n\nEpoch: {}\n'.format(actual_epoch))
        try:
            MODEL.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=EPI)
        except OSError:
            # sometimes an OSError occurs, especially on Windows 10 Systems...
            print('Encountered an Error while training this epoch')
            continue

        actual_epoch += EPI
        txt = generate_text(MODEL, GENERATE_LENGTH, ix_to_char)
        with open('generated_' + str(actual_epoch), 'wb') as out:
            out.write(txt.encode(errors='ignore'))
        MODEL.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, actual_epoch))

# performing generation only
elif MODE == 'test':
    if WEIGHTS == '':
        print('Specify saved Weights!')
        exit(1)

    generate_text(MODEL, GENERATE_LENGTH, ix_to_char)
    print('\n\n')

else:
    print('\n\nNothing to do!')
