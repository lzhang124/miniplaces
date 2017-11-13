import datetime
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from DataLoader import *


def accuracy5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


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

if __name__ == "__main__":
    batch_size = 64
    load_size = 256
    fine_size = 224
    c = 3
    data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
    lr = 0.1
    training_iters = 50000
    step_display = 50
    step_save = 10000
    path_save = 'vgg_{}.h5'
    start_from = None

    opt_data_train = {
        'data_root': '../../data/images/',
        'data_list': '../../data/train.txt',
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': True
        }
    opt_data_val = {
        'data_root': '../../data/images/',
        'data_list': '../../data/val.txt',
        'load_size': load_size,
        'fine_size': fine_size,
        'data_mean': data_mean,
        'randomize': False
        }

    loader_train = DataLoaderDisk(**opt_data_train)
    loader_val = DataLoaderDisk(**opt_data_val)

    if start_from is not None:
        model = load_model(path_save.format(start_from))
    else:
        model = VGG_16()
        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[categorical_accuracy, accuracy5])


    model.train_on_batch

    step = 0
    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        labels_batch = np.eye(100)[labels_batch.astype(int)]
        l, acc1, acc5 = model.train_on_batch(images_batch, labels_batch)

        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%H:%M:%S")))

            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)
            labels_batch_val = np.eye(100)[labels_batch_val.astype(int)]
            vl, vacc1, vacc5 = model.test_on_batch(images_batch_val, labels_batch_val)
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.6f}".format(vl) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(vacc1) + ", Top5 = " + \
                  "{:.4f}".format(vacc5))

        step += 1

        # Save model
        if step % step_save == 0:
            model.save(path_save.format(step))
            print("Model saved at Iter %d !" %(step))

    print("Optimization Finished!")

    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)
        labels_batch = np.eye(100)[labels_batch.astype(int)]
        l, acc1, acc5 = model.test_on_batch(images_batch, labels_batch)
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
