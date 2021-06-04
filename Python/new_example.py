import argparse, sys, time, csv, time
import numpy as np
import blockset
from blockset import external, ensemble
from sklearn import datasets
from sklearn.ensemble import *

num_trees = 100
sklearn_save_filename = 'iris_sklearn_model.json'
blockset_save_filename = 'iris_blockset_model.json'

iris = datasets.load_iris()
sklearn_rf_model = RandomForestClassifier(n_estimators=num_trees, n_jobs=-1)
sklearn_rf_model.fit(iris.data, iris.target)


external.write_to_json(sklearn_rf_model, sklearn_save_filename)

model = ensemble.BlocksetBase()
model.initRandomForestClassifier()
model.loadJSONModel(sklearn_save_filename)
model.pack()
model.serialize(blockset_save_filename)

