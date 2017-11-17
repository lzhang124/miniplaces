import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.utils.np_utils import to_categorical
from DataLoader import *


def VGG_16():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='softmax'))

    return model


def create_generator(loader, batch_size):
    loader.reset()
    while True:
        yield loader.next_batch(batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', default=25, type=int)
    parser.add_argument('-e', default=12, type=int)
    parser.add_argument('-l', '--load', default=False, action='store_true')
    parser.add_argument('-f', '--file', default='vgg19_bn.h5')
    parser.add_argument('-v', '--val', default=True, action='store_false')
    parser.add_argument('-t', '--test', default=True, action='store_false')
    args = parser.parse_args()
    
    batch_size = args.b
    epochs = args.e
    load_size = 256
    fine_size = 224
    lr = 0.0001
    data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
    path_save = args.file
    load = args.load

    opt_data_train = {
        'data_root': '../../data/images/',
        'data_list': '../../data/train.txt',
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': True,
        'num_categories': 100,
        'labels': True
        }
    opt_data_val = {
        'data_root': '../../data/images/',
        'data_list': '../../data/val.txt',
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False,
        'num_categories': 100,
        'labels': True
        }
    opt_data_test = {
        'data_root': '../../data/images/',
        'data_list': '../../data/test.txt',
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False,
        'num_categories': 100,
        'labels': False
        }

    loader_train = DataLoaderDisk(**opt_data_train)
    loader_val = DataLoaderDisk(**opt_data_val)
    loader_test = DataLoaderDisk(**opt_data_test)

    assert loader_val.size() % batch_size == 0, 'Batch size must be a divisor of {}'.format(loader_val.size())
    steps_per_epoch = loader_train.size() / batch_size
    validation_steps = loader_val.size() / batch_size
    test_steps = loader_test.size() / batch_size

    if load:
        model = load_model(path_save)
    else:
        model = VGG_16()
        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[categorical_accuracy, top_k_categorical_accuracy])

        checkpoint = ModelCheckpoint(path_save)
        callbacks_list = [checkpoint]
        model.fit_generator(
            generator=create_generator(loader_train, batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks_list,
            validation_data=create_generator(loader_val, batch_size),
            validation_steps=validation_steps
        )
        model.save(model_file)
        print 'Training Finished!'

    if args.val:
        print 'Validating...'
        loader_val.reset()
        l, acc1, acc5 = model.evaluate_generator(
            generator=create_generator(loader_val, batch_size),
            steps=validation_steps
        )
        print 'loss: {}, acc1: {}, acc5: {}'.format(l, acc1, acc5)

    if args.test:
        print 'Predicting...'
        preds = model.predict_generator(
            generator=create_generator(loader_test, batch_size),
            steps=test_steps,
            verbose=1
        )

        print 'Saving predictions...'
        with open('../../data/test.txt','r') as lines:
            filenames = [line.split(' ')[0] for line in lines]

        with open('../../evaluation/test.pred.txt','w') as file:
            top_indices = preds.argsort()[:,-1:-6:-1]
            for i in xrange(len(preds)):
                top5 = ' '.join(str(j) for j in top_indices[i])
                file.write(filenames[i] + ' ' + top5 + '\n')
