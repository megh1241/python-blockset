import argparse, sys, time, csv, time
import numpy as np
import blockset
from blockset import external, ensemble
from sklearn import datasets
from sklearn.ensemble import *
from sklearn.metrics import *


num_trees = 100
sklearn_save_filename = 'iris_sklearn_model.json'
blockset_save_filename = 'iris_blockset_model.json'
iris = datasets.load_iris()
'''
print(iris.target)
sklearn_rf_model = RandomForestClassifier(n_estimators=num_trees, n_jobs=-1)
sklearn_rf_model.fit(iris.data, iris.target)


external.write_to_json(sklearn_rf_model, sklearn_save_filename)

model = ensemble.BlocksetBase()
model.initRandomForestClassifier()

model.loadJSONModel(sklearn_save_filename)
model.pack()
model.serialize(blockset_save_filename)

'''
model = ensemble.BlocksetBase()
model.initRandomForestClassifier()
model.loadBlocksetModel(blockset_save_filename)
#print(iris.data[0])
iter1=0
wrong = 0
for i in iris.data:
    labels = model.predict(i)
    print(labels)
    print(iris.target[iter1])
    if labels != iris.target[iter1]:
        wrong+=1
    iter1+=1
print('wrong: ')
print(wrong)

#score = accuracy_score(iris.target, labels)
#print(score)
