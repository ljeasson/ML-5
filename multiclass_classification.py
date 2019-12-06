import numpy as np
from helpers import generate_training_data_multi
from binary_classiﬁcation import svm_train_brute

# Use svm_train_brute() to train one binary classifier 
# for each class.  Return Y decision boundaries, one
# for each class.  W is array of ws, and B is array of bs
def svm_train_multiclass(training_data):
    W = 1
    B = 0
    return [W, B]

# Take C decision boundaries as input and test point x
# and return predicted class c.
# No special cases to account for.
# Returns -1 (null) when test point belongs to no class
def svm_test_multiclass(W, B, x):
    return c