model1 = load_model('vgg_50000.h5')
model2 = load_model('vgg_50000.h5')
batch_size = 50;

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

loader_test = DataLoaderDisk(**opt_data_test)

def create_generator(loader, batch_size):
    loader.reset()
    while True:
        yield loader.next_batch(batch_size)

print 'Predicting...'
preds1 = model1.predict_generator(
    generator=create_generator(loader_test, batch_size),
    steps=test_steps,
    verbose=1
)
preds2 = model2.predict_generator(
    generator=create_generator(loader_test, batch_size),
    steps=test_steps,
    verbose=1
)

assert len(preds1) == len(preds2)

print 'Saving predictions...'
with open('../../data/test.txt','r') as lines:
    filenames = [line.split(' ')[0] for line in lines]

with open('../../evaluation/averaged.txt','w') as file:
    top_indices1 = preds1.argsort()[:,-1:-6:-1]
    top_indices2 = preds2.argsort()[:,-1:-6:-1]
    top_indices = np.around((top_indices1+top_indices2)/2)
    for i in xrange(len(preds1)):
        top5 = ' '.join(str(j) for j in top_indices[i])
        file.write(filenames[i] + ' ' + top5 + '\n')
