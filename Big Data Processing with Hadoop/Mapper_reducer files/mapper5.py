#!/usr/bin/python3
import fileinput
import re
import sys
import os
import csv
import pandas as pd
import numpy as np


#read test data
test=pd.read_csv("./knn/Test.csv")
X_test=test.iloc[:,0:48].values

#read each sample from training data
for line in csv.reader(iter(sys.stdin.readline,'')):
    if line:

        #store the label of the each training sample
        label=line.pop()
        line=list(map(float,line))


        #if the record is not header find distances to each test sample
        if sorted(line)!=list(np.arange(min(line),max(line)+1.0,1.0)):  
            X_train=np.array(line)
            X_train=X_train.reshape((1,48))
            distances = np.sqrt(np.sum(X_train**2, axis=1) + np.sum(X_test**2, axis=1)[:, np.newaxis] - 2 * np.dot(X_test, X_train.T)) 
            test_list=X_test.tolist()

            #Append distance, label to each test sample <testsample,distanc,label>
            for i,j in enumerate(test_list):
                test_list[i].append(distances[i][0])
                test_list[i].append(label)

            #Write to stream
            for _ in test_list:
                data=list(map(str,_))
                print(",".join(data))

