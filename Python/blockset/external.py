from sklearn.ensemble import *
import csv, json, sys, os


def argmax_1(a):
    return max(range(len(a)), key=lambda x: a[x])


def write_to_json(model1, filename, regression=False):
    final_count = 0
    new_dict = {'estimators': {'nodes': [], 'values': [] } }
    for count, estimator in enumerate(model1.estimators_):
        nodes = estimator.tree_.__getstate__()['nodes'].tolist()
        newnodes = [list((i[0], i[1], i[2], i[3], i[5])) for i in nodes]
        length = len(nodes)
        values = estimator.tree_.__getstate__()['values']
        for i in range(length):
            if newnodes[i][0] == -1:
                print(values[i])
                newnodes[i][3] = values[i][0][0] 
        new_dict['estimators']['nodes'].append(newnodes)
        final_count += 1
    if regression:
        new_dict['n_classes'] = -1
    else:
        new_dict['n_classes'] = model1.n_classes_

    new_dict['n_estimators'] = final_count
    json_obj = json.dumps(new_dict)
    with open(filename, "w") as outfile:
        outfile.write(json_obj)


def write_to_json_gbt(model, filename, regression=False):
    new_dict = {'estimators': {'nodes': [], 'values': [] } }
    final_count = 0
    for count, estimator_list in enumerate(model.estimators_):
        estimator = estimator_list[0]
        nodes = estimator.tree_.__getstate__()['nodes'].tolist()
        newnodes = [[i[0], i[1], i[2], i[3], i[5]] for i in nodes]
        length = len(nodes)
        values = estimator.tree_.__getstate__()['values']
        for i in range(length):
           
            if newnodes[i][0] == -1:
                #print('leaf!')
                #print(values[i])
                newnodes[i][3] = values[i][0][0]
                #print(newnodes[i][3])
                #newnodes[i][2] = argmax_1(list(values[i][0]))
        final_count += 1
        new_dict['estimators']['nodes'].append(newnodes)

    if regression:
        new_dict['n_classes'] = -1
    else:
        new_dict['n_classes'] = model.n_classes_

    new_dict['n_estimators'] = final_count
    json_obj = json.dumps(new_dict)
    with open(filename, "w") as outfile:
        outfile.write(json_obj)
    if regression:
        import joblib
        filename = filename[:-5]
        filename = 'init' + filename + '.joblib'
        joblib.dump(model.init_, filename)
