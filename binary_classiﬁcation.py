import numpy as np
import helpers

def generate_training_data_binary(num):
    return

def plot_training_data_binary(data):
    return

data = generate_training_data_binary(num)
plot_training_data_binary(data)

# Helper function that computes distance between a point
# and a hyperplane (a line in 2D) 
# DON'T WORRY ABOUT HIGHER DIMENSIONS!
def distance_point_to_hyperplane(pt, w, b):
    return

# Takes a set of data and a separator
# and computes the margin
def compute_margin(data, w, b):
    return

# Returns decision boundary for classifier
# Line characterized by w and b that separates
# training data with largest margin
def svm_train_brute(training_data):
    dist = distance_point_to_hyperplane(pt, w, b)
    return

# Test new data given a decision boundary
def svm_test_brute(w, b, x):
    return


[w,b,S] = svm_train_brute(training_data)
margin = compute_margin(data, w, b)
y = svm_test_brute(w,b,x)

