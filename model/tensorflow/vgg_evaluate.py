import datetime
import numpy as np
from keras.models import Sequential, load_model
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.utils.np_utils import to_categorical
from DataLoader import *

load_size = 256
fine_size = 224
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# file = open("../../data/test.txt","w") 
# lines = open("../../data/val.txt", "r")
# for line in lines:
# 	n = line.split("/")[1].split(" ")[0]
# 	file.write("test/" + n + " " + str(0) + "\n") 
# file.close() 

lines = open("../../data/test.txt","r")
filenames = [line.split(" ")[0] for line in lines]

opt_data_test = {
    'data_root': '../../data/images/',
    'data_list': '../../data/test.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'num_categories': 100
    }

loader_test = DataLoaderDisk(**opt_data_test)
test_steps = loader_test.size() / batch_size
preds = model.predict_generator(
    generator=create_generator(loader_test, batch_size),
    steps=test_steps
)

file = open("test.pred.txt","w") 
for i,pred in enumerate(preds):
    top_indices = pred.argsort()[-5:][::-1]
    top5 = " ".join(str(i) for i in top_indices)
    file.write(filenames[i] + " " + top5 + "\n") 
file.close() 

# images_batch_test, labels_batch_test = loader_test.next_batch(loader_test.size())
# model = load_model('vgg16_bn.h5')
# preds = model.predict(images_batch_test)

