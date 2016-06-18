'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import sys
import time
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

sgd_learning_rate = 0.05
nb_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[0:6000]
y_train = y_train[0:6000]
X_test = X_test[0:1000]
X_test_plot = X_test
y_test = y_test[0:1000]

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# load the model to predict.
from keras.models import model_from_json
fnameMODEL = 'mnist_cnn_step5_save.py'
model_f = fnameMODEL + '.json'
weight_f = fnameMODEL + '.hdf5'

reload_model = model_from_json(open(model_f).read())
reload_model.load_weights(weight_f)
sgd = SGD(lr=sgd_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
reload_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

score = reload_model.evaluate(X_test, Y_test, verbose=0)
print('Reload Test score:', score[0])
print('Reload Test accuracy:', score[1])

while True:
    print('Please input the index of the test sample:0~'+str(len(X_test)-1))
    test_idx_str = sys.stdin.readline()
    try:
        test_idx = int(test_idx_str)
        if test_idx < 0 or test_idx >= len(X_test):
            raise Exception
    except Exception as e:
        print('Please input the integer in the index range.')
        break
    img2d = X_test_plot[test_idx]
    GREYSCALE = ['  ', '..', '::', '++', '##', '@@']
    print('X_test['+str(test_idx)+']')
    print('\n'.join([''.join([GREYSCALE[int(float(byte)*len(GREYSCALE)/256)] for byte in line]) for line in img2d]))

    start_t1 = time.time()
    result = reload_model.predict(X_test[test_idx].reshape(1, 1, img_rows, img_cols))[0]
    start_t2 = time.time()
    print('predict result:')
    for i in xrange(len(result)):
        print(str(i)+' possibility: %0.6f' %(float(result[i])))
    print('predict time:'+str(start_t2-start_t1))
