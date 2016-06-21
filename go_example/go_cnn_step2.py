'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.optimizers import SGD

DEEP_MaxEpochInTraining = 0
class EarlyStopping(Callback):
    '''Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In 'min' mode,
            training will stop when the quantity
            monitored has stopped decreasing; in 'max'
            mode it will stop when the quantity
            monitored has stopped increasing.
    '''
    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto'):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.count = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
    
        global DEEP_MaxEpochInTraining
        self.count = self.count+1
        DEEP_MaxEpochInTraining = max(DEEP_MaxEpochInTraining,self.count)
    
        current = logs.get(self.monitor)
        if current is None:
            print('Early stopping requires %s available!' % (self.monitor))
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1    

DEEP_EARLY_STOP_P = 10
DEEP_SGDLR = 0.1
early_stopping = EarlyStopping(monitor='val_acc', patience=DEEP_EARLY_STOP_P)

batch_size = 200
nb_classes = 361
nb_epoch = 200

# input image planes
img_planes = 2
# input image dimensions
img_rows, img_cols = 19, 19
# number of convolutional filters to use
nb_filters = 48
# size of pooling area for max pooling
#nb_pool = 2
# convolution kernel size
nb_conv = 7

# the data, shuffled and split between train and test sets
def load_npy(Xfilename, yfilename):

    total_stat_list = np.load(Xfilename)
    total_next_list = np.int8(np.load(yfilename))
    rval = total_stat_list, total_next_list
    return rval

filename = 'test_750test.dat_'
Xfilename = filename + 'X.npy'
yfilename = filename + 'Y.npy'

X_data, y_data = load_npy(Xfilename, yfilename)
print('X_data.shape:' + str(X_data.shape))
print('X_data[0]:' + str(X_data[0]))
print('y_data.shape:' + str(y_data.shape))
print('y_data[0]:' + str(y_data[0]))
total_len = len(y_data)
X_train = X_data[0:int(total_len*0.8)]
y_train = y_data[0:int(total_len*0.8)]
X_test = X_data[int(total_len*0.8):]
y_test = y_data[int(total_len*0.8):]

X_train = X_train.reshape(X_train.shape[0], img_planes, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], img_planes, img_rows, img_cols)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_train[0]:' + str(Y_train[0]))
print(Y_train.shape[0], 'train samples')
print(Y_test.shape[0], 'test samples')

