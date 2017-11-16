with open('../../evaluation/val.pred.txt','r') as f:
    preds = {}
    for line in f:
        pred = line.split()
        preds[pred[0]] = pred[1:]
with open('../../data/val.txt','r') as f:
    golds = {}
    for line in f:
        gold = line.split()
        golds[gold[0]] = gold[1]

print preds
print gold
count = 0
assert len(preds) == len(golds)
for file in preds:
    if golds[file] in preds[file]:
        print golds[file]
        print preds[file]
        count+=1
print float(count)/len(preds)
