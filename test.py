from helpers import generate_training_data_binary, plot_training_data_binary, generate_training_data_multi, plot_training_data_multi, plot_training_data_binary_lines
from binary_classification import svm_train_brute, svm_test_brute
from multiclass_classification import svm_train_multiclass, svm_test_multiclass 

for num in (1,2,3,4):
    # Binary Classification
    data = generate_training_data_binary(num)
    #plot_training_data_binary(data)

    [w,b,S] = svm_train_brute(data)
    print(w,b,S)
    plot_training_data_binary_lines(data,S)
    
'''
for num in (1,2):
    # Multiclass Classification
    [data, Y] = generate_training_data_multi(num)
    plot_training_data_multi(data)

    [W, B] = svm_train_multiclass(data)
'''