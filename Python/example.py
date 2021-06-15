import argparse, sys, time, csv, time
import numpy as np
import blockset
from blockset import external, ensemble
from sklearn import datasets
from sklearn.ensemble import *
from sklearn.metrics import *
import joblib



num_trees = 100
sklearn_save_filename = 'iris_sklearn_model.json'
blockset_save_filename = 'iris_blockset_model.json'

#Step 1: load data
iris = datasets.load_iris()

#Step 2: Train a Gradient boosting / Random forest model
sklearn_rf_model = GradientBoostingClassifier(n_estimators=num_trees)
sklearn_rf_model.fit(iris.data, iris.target)

#Step 3: Use the helper method in Blockset's external module to convert the sklearn model to json and save to file.
external.write_to_json_gbt(sklearn_rf_model, sklearn_save_filename, regression=False)

#Step 4: initialize Blockset model
model = ensemble.BlocksetBase()
model.initGradientBoostingClassifier()

#Step 5: Load the saved sklearn json model
model.loadJSONModel(sklearn_save_filename)

#Step 6: Pack the model
model.pack()

#Step 7: Serialize the model to disk in a custom binary format
model.serialize(blockset_save_filename)

#Step 8: Load the Blockset Model
model.loadBlocksetModel(blockset_save_filename)
label_vec = []
for i in iris.data:
    #Predict labels/targets for test data
    label_vec.append(model.predict(i))

print(label_vec)
print(sklearn_rf_model.predict(iris.data))
print(iris.target)
