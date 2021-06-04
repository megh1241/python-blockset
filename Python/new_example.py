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
sklearn_rf_model = GradientBoostingClassifier(n_estimators=num_trees)
sklearn_rf_model.fit(iris.data, iris.target)


external.write_to_json_gbt(sklearn_rf_model, sklearn_save_filename)

model = ensemble.BlocksetBase()
model.initGradientBoostedClassifier()

model.loadJSONModel(sklearn_save_filename)
model.pack()
model.serialize(blockset_save_filename)

'''
model = ensemble.BlocksetBase()
model.initGradientBoostedClassifier()
model.loadBlocksetModel(blockset_save_filename)
label_vec = []
label_vec = model.predict(iris.data)
print(label_vec)
print(iris.target)
score = accuracy_score(iris.target, label_vec)
print("accuracy score: ", end="")
print(score)
