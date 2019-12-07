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
    num_samples = len(training_data)
    w = np.zeros(num_samples)
    w = 0
    b = 0
    S = []
    
    l_rate = 1
    epoch = 100000
    
    '''
    for sample in range(num_samples):
        pt = (training_data[sample][0], training_data[sample][1])
        dist = distance_point_to_hyperplane(pt, w, b)

    margin = compute_margin(training_data, w, b)
    
    '''
    for e in range(epoch):
        for i, val in enumerate(training_data):
            val1 = np.dot((training_data[i][0]), w)
            #val2 = np.dot(training_data[i][1], w)
            
            if (training_data[i][2]*val1 < 1):
                w = w + l_rate * ((training_data[i][2] * training_data[i][0]) - (2*(1/epoch)*w))
            else:
                w = w + l_rate * (-2*(1/epoch)*w)
            
    for i, val in enumerate(training_data):
        S.append(np.dot(training_data[i][0], w))

    S = np.array(S)
    return w,b,S

# Test new data given a decision boundary
def svm_test_brute(w, b, x):
    y = svm_test_brute(w,b,x)
    return