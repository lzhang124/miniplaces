from operator import add
import numpy as np

with open('../../evaluation/vgg16.val.pred.txt','r') as f:
    preds = {}
    for line in f:
        pred = line.split()
        preds[pred[0]] = pred[1:]
with open('../../evaluation/demo.val.pred.txt','r') as f:
    preds2 = {}
    for line in f:
        pred = line.split()
        preds2[pred[0]] = pred[1:]
        
assert len(preds) == len(preds2)
names = preds.keys()
names.sort()

with open('../../evaluation/averaged.txt','w') as f:
    for i in xrange(len(preds)):
        img = names[i]
        p1 = np.array(map(float, preds[img]))
        p2 = np.array(map(float, preds2[img]))
        new_preds = np.around((p1+p2)/2)
        new_preds = new_preds.astype(int).tolist()
        top5 = ''
        for i in range(5):
            top5 += str(new_preds[i]) + " "
        f.write(names[i] + ' ' + top5 + '\n')