from helpers import generate_training_data_binary, plot_training_data_binary, generate_training_data_multi, plot_training_data_multi, plot_training_data_binary_lines
from binary_classification import svm_train_brute, svm_test_brute
from multiclass_classification import svm_train_multiclass, svm_test_multiclass 
import numpy as np

test_data = np.array([
    [-2, 1, 1],
    [4, 3, -1],
    [1, 3, -1],
    [-1, 1, 1],
    [2, -1, -1]
])

for num in (1,2,3,4):
    # Binary Classification
    data = generate_training_data_binary(num)
    
    # Plot original training data
    #plot_training_data_binary(data)

    # Train Binary classification
    [w,b,S] = svm_train_brute(data)
    print(w,b,S)
    
    # Training data 
    training_acc = 0
    correct, count = 0, 0
    for x in data:
        y = svm_test_brute(w,b,x)
        if x[2] == y:
            correct += 1
        count += 1
    training_acc = correct / count 
    print("Training Acc:",training_acc)

    # Testing data
    testing_acc = 0
    correct, count = 0, 0
    for x in test_data:
        y = svm_test_brute(w,b,x)
        if x[2] == y:
            correct += 1
        count += 1
    testing_acc = correct / count 
    print("Testing Acc:",testing_acc,"\n")

    # Plot training data with decision boundaries
    plot_training_data_binary_lines(data,w,b,S)

'''
for num in (1,2):
    # Multiclass Classification
    [data, Y] = generate_training_data_multi(num)
    plot_training_data_multi(data)

    [W, B] = svm_train_multiclass(data)
'''