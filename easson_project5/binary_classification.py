import numpy as np

# Helper function that computes distance between a point
# and a hyperplane (a line in 2D) 
# DON'T WORRY ABOUT HIGHER DIMENSIONS!
def distance_point_to_hyperplane(pt, w, b):
    p1=np.array([0.0, (w*0.0)+b])
    p2=np.array([15.0,(w*15.0) + b])
    p3=np.array([pt[0],pt[1]])
    
    d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    return d

# Takes a set of data and a separator
# and computes the margin
def compute_margin(data, w, b):
    # Separate into (+) and (-) samples
    distances_pos = {}
    distances_neg = {}
    label = data[0][2]

    for d in data:
        if d[2] == label:
            distances_pos.update({(d[0], d[1]) : 0})
        else:
            distances_neg.update({(d[0], d[1]) : 0})

    # Get distances for (+) and (-) samples
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
    return w


# Returns decision boundary for classifier
# Line characterized by w and b that separates
# training data with largest margin
def svm_train_brute(training_data):
    w = np.zeros(training_data.shape[1]-1)
    w = 0.0
    b = 0.0
    S = []

    margin = compute_margin(training_data, w, b)
    w = margin
    S.append(margin)
    
    S = np.array(S)
    return w,b,S

# Test new data given a decision boundary
def svm_test_brute(w, b, x):
    pred = 0
    result = np.dot(x, w)

    if result[0] >= 0:
        pred = 1
    else:
        pred = -1
    return pred