import numpy as np

# Helper function that computes distance between a point
# and a hyperplane (a line in 2D) 
# DON'T WORRY ABOUT HIGHER DIMENSIONS!
def distance_point_to_hyperplane(pt, w, b):
    p1=np.array([0, (w*0)+b])
    p2=np.array([15,(w*15) + b])
    p3=np.array([pt[0],pt[1]])
    
    d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    return d

# Takes a set of data and a separator
# and computes the margin
def compute_margin(data, w, b):
    return


# Returns decision boundary for classifier
# Line characterized by w and b that separates
# training data with largest margin
def svm_train_brute(training_data):
    w = 1
    b = 0
    S = np.empty([2, 2])
    
    num_samples = training_data.shape[0]
    for sample in range(num_samples):
        pt = (training_data[sample][0], training_data[sample][1])
        dist = distance_point_to_hyperplane(pt, w, b)
        print(dist)
    print()

    margin = compute_margin(training_data, w, b)
    
    #print(w,b,"\n",S,"\n")
    return w,b,S

# Test new data given a decision boundary
def svm_test_brute(w, b, x):
    y = svm_test_brute(w,b,x)
    return