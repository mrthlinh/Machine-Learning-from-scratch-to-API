import numpy as np
def load_data(filename):
    input = np.loadtxt(filename, dtype='d', delimiter=',')
    row_length,col_length = input.shape

    row_training  = int(row_length) #336
    training_data = input[0:row_training,0:col_length]

    # Initialize training
    train_x = input[:,1:col_length]
    train_y = input[:,0:1]
    return [train_x,train_y]
def KNN(filename,k):
    train_x , train_y = load_data("wdbc_train.data")
    test_x  , test_y  = load_data(filename)
    
    
    row_length = test_x.shape[0]
    d = {}    
    f = np.empty(test_y.shape) #Initialize an empty numpy array
    
    # Iterate through all data in test set
    for i in range(0,row_length):
#     for i in range(0,1):
        single_x = test_x[i]
#         x_ = np.delete(x,i,0) #remove selected x
        list_point = np.sqrt(np.sum((np.square(train_x - single_x)), axis = 1))
        
        # Create a dictionary with key = index , value = distance 
        d = dict(enumerate(list_point.flatten(), 0))
        
        # Sort the dictionary based on their keys
        d = sorted(d.items(), key=lambda x: x[1])
        
        check = 0
        for j in range(0,k):
            index = d[j][0]
#             print ("result at index "+str(index) +" is "+str(y[index]))
#             print("point "+str(j)+": "+str(d[j][0]))
            check = check + train_y[index]
#         print ("result at that point is "+str(check))
        if (check > 0):
            f[i] = 1
        else:
            f[i] = -1
    
    result = f * test_y
    result[result < 0] = 0
    return (100*np.sum(result) / row_length)

# The result
list_k =[1,5,11,15,21,27,35,47,55]
for k in list_k:
    print ("K = "+str(k))
    print ("Accuracy: "+str(KNN("wdbc_train.data",k)))