import numpy as np
from helpers import generate_training_data_multi
from binary_classiﬁcation import svm_train_brute, distance_point_to_hyperplane

# Takes a set of data, a separator, and a label
# and computes the margin
def compute_margin(data, w, b, label):
    # Separate into label and not label samples
    distances_pos = {}
    distances_neg = {}

    for d in data:
        if d[2] == label:
            distances_pos.update({(d[0], d[1]) : 0})
        else:
            distances_neg.update({(d[0], d[1]) : 0})

    # Get distances for label and not label samples
    for pos in distances_pos:
        dist = distance_point_to_hyperplane(pos, w, b)
        distances_pos[pos] = abs(dist)

    for neg in distances_neg:
        dist = distance_point_to_hyperplane(neg, w, b)
        distances_neg[neg] = abs(dist)

    # Get min distance from each list
    distances_pos = dict(sorted(distances_pos.items(), key=lambda x: x[1]))
    distances_neg = dict(sorted(distances_neg.items(), key=lambda x: x[1]))
    minimum_pos = min(distances_pos, key=distances_pos.get)
    minimum_neg = min(distances_neg, key=distances_neg.get)

    # Get perpendicular line
    w = np.cross(minimum_pos,minimum_neg)
    return w,b

# Use svm_train_brute() to train one binary classifier 
# for each class.  Return Y decision boundaries, one
# for each class.  W is array of ws, and B is array of bs
def svm_train_multiclass(training_data):
    W = []
    B = []
    labels = []
    for i in range(len(training_data)): labels.append(training_data[i][2])
    Y = len(set(labels))
    labels = list(set(labels))

    # Compute margin
    for i in range(Y):
        w = 0.0
        b = 0.0
        w,b = compute_margin(training_data,w,b,labels[i])
        W.append(w)
        B.append(b)

    W = np.array(W)
    B = np.array(B)
    return [W, B]

# Take C decision boundaries as input and test point x
# and return predicted class c.
# No special cases to account for.
# Returns -1 (null) when test point belongs to no class
def svm_test_multiclass(W, B, x):
    c = -1
    return c