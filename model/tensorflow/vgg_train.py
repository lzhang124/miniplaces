import datetime
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.utils.np_utils import to_categorical
from DataLoader import *


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def create_generator(loader, batch_size):
    while True:
        yield loader.next_batch(batch_size)


if __name__ == '__main__':
    batch_size = 25
    load_size = 256
    fine_size = 224
    c = 3
    data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
    lr = 0.0001
    epochs = 500
    step_display = 50
    step_save = 10000
    path_save = 'vgg_bn.h5'
    load = False

    opt_data_train = {
        'data_root': '../../data/images/',
        'data_list': '../../data/train.txt',
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': True,
        'num_categories': 100
        }
    opt_data_val = {
        'data_root': '../../data/images/',
        'data_list': '../../data/val.txt',
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False,
        'num_categories': 100
        }

    loader_train = DataLoaderDisk(**opt_data_train)
    loader_val = DataLoaderDisk(**opt_data_val)

    assert loader_val.size() % batch_size == 0, "Batch size must be a divisor of {}".format(loader_val.size())
    steps_per_epoch = loader_train.size() / batch_size
    validation_steps = loader_val.size() / batch_size

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

    loader_val.reset()
    model.evaluate_generator(
        generator=create_generator(loader_val, batch_size),
        steps=validation_steps
    )
