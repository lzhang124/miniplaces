pred = open('../../evaluation/val.pred.txt','r')
preds = [line.split()[1:] for line in pred]
gold = open('../../data/val.txt','r')
golds = [line.split()[1] for line in gold]

count = 0
print(len(preds))
print(len(golds))
for i in range(len(preds)):
    if golds[i] in preds[i]:
        print(golds[i])
        print(preds[i])
        count+=1
print(float(count)/len(preds))