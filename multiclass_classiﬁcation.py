import numpy as np
import helpers
from binary_classiﬁcation import svm_train_brute

def generate_training_data_multi(num):
    return

def plot_training_data_multi(data):
    return

[data, Y] = generate_training_data_multi(num)
plot_training_data_multi(data)


# Use svm_train_brute() to train one binary classifier 
# for each class.  Return Y decision boundaries, one
# for each class.  W is array of ws, and B is array of bs
def svm_train_multiclass(training_data):
    return [W, B]

# Take C decision boundaries as input and test point x
# and return predicted class c.
# No special cases to account for.
# Returns -1 (null) when test point belongs to no class
def svm_test_multiclass(W, B, x):
    return c

[W, B] = svm_train_multiclass(training_data)
y = svm_test_multiclass(W, B, x)