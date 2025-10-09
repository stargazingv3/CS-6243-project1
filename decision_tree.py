# Zijie Zhang, Sep 2025

import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

import inspect # DEBUG
PRINT_DEBUG = False
def DEBUG(*args, **kwargs):
    if PRINT_DEBUG:
        print(" . " * len(inspect.stack()), *args, **kwargs)

# Entropy and Information Gain
def entropy(y):
    # Calculate Information Entropy for Distribution y
    # takes an array (y), gives 1 number (entropy). I think this is the log2 function thing
    # information gain/entropy = -p(x)log2(p(x)) summed over all possible values of the variable
    # ex. -.1*log2(.1) - .9*log2(.9)
    if len(y) == 0:
        return 0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, splits):
    # Calculate Information Gain, given a distribution y and a set of indicies splits
    # array of arrays or smtn. array of arrays of indicies for splits?
    # ah, so its like "I'll make a list out of the og one like this: first split has index 0,1,8 from og list, second has 2,3,9, and third has 4,5,6,7. what is the info gain if I split like this?"
    # so return just info gain number
    y_splits = [ [] for i in range(len(splits)) ]
    for i in range(len(splits)):
        for index in splits[i]:
            y_splits[i].append(float(y[index]))
    children_entropy = 0
    for y_split in y_splits:
        children_entropy += (len(y_split) / sum([len(x) for x in y_splits])) * entropy(y_split)

    parent_entropy = entropy(y)

    info_gain = parent_entropy - children_entropy

    return info_gain

def majority_label(y):
    if list(y) == []:
        print("ERROR: list y is empty, cannot make prediction")
        return -1
    return Counter(y).most_common(1)[0][0]

# Finding the best split
def best_split(X, y, split_features):
    DEBUG("")
    DEBUG("best_split() entry")
    n, m = X.shape # height, width
    best_gain, best_feature, best_threshold, best_splits = -1, None, None, None

    for j in range(m): # for column (feature) index in <num collumns>
        DEBUG("")
        if AttributeNames[j] in split_features:
            DEBUG("we already did this feature (", AttributeNames[j], "), skipping...")
            continue
        DEBUG("hey were doing category", AttributeNames[j], "rn btw")
        col = X[:, j]
        DEBUG("col is", col)
        info_gain = -1
        thresh = None
        splits = None
        try: # NUMERICAL DATA
            col = col.astype(float)
            values = list(set(col.tolist()))
            values.sort()
            DEBUG("values are", values)
            for value in values:
                num_splits = [[], []]
                for i in range(len(col)): # split into  <= or > thresh
                    if col[i] <= value:
                        num_splits[0].append(i)
                    else:
                        num_splits[1].append(i)
                num_gain = information_gain(y, num_splits)
                if num_gain > info_gain:
                    info_gain = num_gain
                    thresh = value
                    splits = num_splits

        except ValueError as e: # CATEGORICAL DATA (bc we couldnt convert to int)
            # get array of arrays, sub-arrays contain indexes all datapoints with that feature value
            # ie, [[0,1,2],[3,4,5]] means feature values in column are like ['A','A','A', 'B','B','B']
            categories = list(set(col))
            categories.sort()
            DEBUG("categories are", categories)
            splits = [[] for i in range(len(categories))]
            for i in range(len(col)):
                splits[categories.index(col[i])].append(i)
        except Exception as e:
            print("got exception:", e)

        info_gain = information_gain(y, splits)
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = AttributeNames[j]
            best_threshold = thresh
            best_splits = splits
            DEBUG("found new best split: best_gain = ", best_gain, " best_feature = ", best_feature, " best_threshold = ", best_threshold, " best_splits = ", best_splits )
        else:
            DEBUG("not a better split (only", info_gain, ")")


    return best_feature, best_threshold, best_gain, best_splits

# Tree Node
class Node:
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.branches = {}


# Recursive tree fitting
def fit_tree(X, y, depth=0, max_depth=None, split_features=[]):
    DEBUG("fit_tree() entry (depth", depth, ")")


    if len(set(y)) == 1 or len(y) == 0 or (max_depth and depth >= max_depth): # Max Depth or pure node, make leaf
        DEBUG("leaf time yippee we're done")
        node = Node(is_leaf = True, prediction = majority_label(y))
        return node

    # At this point, not a leaf, need to split more

    #Get the best split
    feat, thr, gain, splits = best_split(X, y, split_features)
    if feat is None or gain <= 1e-12: # no good split, just give up here 'cause it aint getting better
        DEBUG("the \"best split\" was mid af so we stopping here")
        node = Node(is_leaf = True, prediction = majority_label(y))
        return node
    DEBUG("best split returned: feat", feat, "thr", thr, "gain", gain, "splits", splits)

    # at this point, we *have* a split that is good enough
    # Todo: Create internal node and recurse for children
    node = Node(feature=feat, threshold=thr)
    DEBUG("RECURSION IS FUN!!! about to make node, and recurse on branches")
    if thr:
        DEBUG("numerical data recursion")
        node.branches = {
            "<=": fit_tree(np.array([X[i] for i in splits[0]]), np.array([y[i] for i in splits[0]]), depth+1, max_depth, split_features),
            ">":  fit_tree(np.array([X[i] for i in splits[1]]), np.array([y[i] for i in splits[1]]), depth+1, max_depth, split_features)
        }
    else:
        DEBUG("categorical data recursion")
        values = list(set(X[:,(AttributeNames.tolist().index(feat))]))
        values.sort()
        node.branches = {value: fit_tree(np.array([X[i] for i in splits[values.index(value)]]), np.array([y[i] for i in splits[values.index(value)]]), depth+1, max_depth, split_features+[feat]) for value in values}

    return node

# Prediction
def predict_one(node, x):
    while not node.is_leaf:
        if node.threshold is not None:
            v = float(x[AttributeNames.tolist().index(node.feature)])
            key = "<=" if v <= node.threshold else ">"
        else:
            key = x[AttributeNames.tolist().index(node.feature)]
        node = node.branches.get(key, None)
        if node is None:
            break
    return node.prediction if node else None

def predict(node, X):
    return np.array([predict_one(node, row) for row in X])

# Printing the tree
def plot_tree(node, attr, depth=0):
    if depth==0:
        print("root")
    if node.is_leaf:
        print(' | '*(depth) + " +-> " + "Predict:", node.prediction)
        return
    if node.threshold is not None:
        print(' | '*(depth) + " +-> " + node.feature + "<=" + str(node.threshold))
        plot_tree(node.branches["<="], attr, depth+1)
        print(' | '*(depth) + " +-> " + node.feature + ">" + str(node.threshold))
        plot_tree(node.branches[">"], attr, depth+1)
    else:
        for key in node.branches.keys():
            print(' | '*(depth) + " +-> " + node.feature + " is " + key)
            plot_tree(node.branches[key], attr, depth+1)

# Test with toy dataset
if __name__ == "__main__":
    filepath = "./data/example.csv"
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
    rawdata = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=object)
    y = rawdata[:, -1]
    X = rawdata[:, :-1]
    AttributeNames = np.array(header[:-1])

    DEBUG(AttributeNames)
    DEBUG(X)
    DEBUG(y)
    tree = fit_tree(X, y, max_depth=3)
    y_pred = predict(tree, X)
    print("Predictions:", y_pred)
    print("Accuracy:", np.mean(y_pred == y))
    plot_tree(tree, AttributeNames)
