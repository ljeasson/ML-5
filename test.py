from helpers import generate_training_data_binary, plot_training_data_binary, generate_training_data_multi, plot_training_data_multi
from binary_classification import svm_train_brute, svm_test_brute
from multiclass_classification import svm_train_multiclass, svm_test_multiclass 

for num in (1,2,3,4):
    # Binary Classification
    data = generate_training_data_binary(num)
    plot_training_data_binary(data)

    [w,b,S] = svm_train_brute(data)
    
for num in (1,2):
    # Multiclass Classification
    [data, Y] = generate_training_data_multi(num)
    plot_training_data_multi(data)

    [W, B] = svm_train_multiclass(data)


'''
import matplotlib.pyplot as plt

datax1=[0,0,0,0,0]
datay1=[1,2,3,4,5]

datax2=[1,1,1,1,1]
datay2=[1,4,9,16,25] 

for i in range(len(datax1)):
    x = (datax1[i], datax2[i])
    y = (datay1[i], datay2[i]) 
    plt.plot(x, y)

plt.show()
'''