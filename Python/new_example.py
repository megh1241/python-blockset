import argparse, sys, time, csv, time
import numpy as np
import blockset
from blockset import external, ensemble
from sklearn import datasets
from sklearn.ensemble import *
from sklearn.metrics import *
import joblib



num_trees = 10
sklearn_save_filename = 'boston_sklearn_model.json'
blockset_save_filename = 'boston_blockset_model.json'
boston = datasets.load_boston()
#iris = datasets.load_iris()
'''
sklearn_rf_model = GradientBoostingRegressor(n_estimators=num_trees, max_depth=4)
sklearn_rf_model.fit(boston.data, boston.target)

external.write_to_json_gbt(sklearn_rf_model, sklearn_save_filename, regression=True)

model = ensemble.BlocksetBase()
model.initGradientBoostingRegressor()

model.loadJSONModel(sklearn_save_filename)
model.pack()
model.serialize(blockset_save_filename)
import joblib
joblib.dump(sklearn_rf_model, 'sk.joblib')

#print(boston.target[0:10])
'''
#sklearn_rf_model = GradientBoostingRegressor(n_estimators=num_trees, max_depth=4)
#sklearn_rf_model.fit(boston.data, boston.target)
sklearn_rf_model = joblib.load('sk.joblib')
model = ensemble.BlocksetBase()
model.initGradientBoostingRegressor()
model.loadBlocksetModel(blockset_save_filename)
label_vec = []
for i in boston.data[0:10]:
    A = []
    A.append(i)
    A = np.array(A).reshape(-1, 1)
    label_vec.append(model.predict(i) )

print(label_vec)
#print(boston.target[0:10])
#label_vec = model.predict(boston.data[0:10])
print(sklearn_rf_model.predict(boston.data[0:10]))
