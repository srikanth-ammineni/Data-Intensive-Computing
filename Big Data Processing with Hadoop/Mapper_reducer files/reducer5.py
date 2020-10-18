#!/usr/bin/python3
import sys
import csv
import numpy as np

prev_test_sample=None
distances=[]
labels=[]

#set k=5
k=5

#read each line from mapper output
for line in csv.reader(iter(sys.stdin.readline,'')):
    line=[x.strip() for x in line]
    curr_test_sample=line[0:48]

    #Find the k shortest distances and predict the final label based on voting
    if curr_test_sample==prev_test_sample or prev_test_sample is None:
        distances.append(line[48])
        labels.append(int(line[49]))
    elif curr_test_sample!=prev_test_sample and prev_test_sample is not None:
        distances_array=np.array(distances)
        labels_array=np.array(labels)
        indices=distances_array.argsort()[:k]
        maxklabels=np.take(labels_array,indices)
        counts=np.bincount(maxklabels)
        pred=np.argmax(counts)

        #Write to stream <testsample,predicted label>
        prev_test_sample.append(str(pred))
        data=list(map(str,prev_test_sample))
        print(",".join(data))

        distances=[]
        labels=[]

    prev_test_sample=curr_test_sample


#Find the predicted label for the final test sample
distances_array=np.array(distances)
labels_array=np.array(labels)
indices=distances_array.argsort()[:k]
maxklabels=np.take(labels_array,indices)
counts=np.bincount(maxklabels)
pred=np.argmax(counts)

#Write to stream <testsample,predicted label>
prev_test_sample.append(str(pred))
data=list(map(str,prev_test_sample))
print(",".join(data))



