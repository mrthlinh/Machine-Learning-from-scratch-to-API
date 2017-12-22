#Final version
import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def load_data(filename):
    input = np.loadtxt(filename, dtype='d', delimiter=',')
    row_length,col_length = input.shape

    row_training  = int(row_length) #336
    training_data = input[0:row_training,0:col_length]

    # Initialize training
    train_x = input[:,1:col_length]
    train_x = np.concatenate((train_x,np.ones((row_training,1))),axis = 1)#Add ones column as bias
    train_y = input[:,0:1]
    return [train_x,train_y]
    
def SVM_primal(filename,c):
    # Load input
    input = np.loadtxt(filename, dtype='d', delimiter=',')
    row_length,col_length = input.shape

    row_training  = int(row_length) #336
    training_data = input[0:row_training,0:col_length]

    # Initialize training
    train_x = input[:,1:col_length]
    train_x = np.concatenate((train_x,np.ones((row_training,1))),axis = 1)#Add ones column as bias
    train_y = input[:,0:1]
    feature_num = train_x.shape[1] #11

    P = np.eye(feature_num-1)
    P = np.concatenate((P,np.zeros((feature_num-1,row_training+1))),axis = 1)
    P = np.concatenate((P,np.zeros((row_training+1,row_training+feature_num))),axis = 0) #347 x 347

    q = c * np.concatenate((np.zeros((1,feature_num)),train_y.T),axis=1)

    h = np.concatenate((-np.ones((row_training,1)),np.zeros((row_training,1))),axis=0)

    x = np.concatenate((train_x,np.zeros((row_training,feature_num))),axis=0) #336 x 347
    eye = np.eye(row_training) # 336 x 336
    eye = np.vstack((eye,eye)) #Add ones column as bias -> already added above
    x = np.concatenate((x,eye),axis=1) #672 x 347
    y = np.concatenate((train_y,train_y),axis=0)
    G = -y * x

    # Quadprop for Primal

    # Define QP parameters (directly)
    P_ = matrix(P)
    q_ = matrix(q.T)
    G_ = matrix(G)
    h_ = matrix(h)

    # Construct the QP, invoke solver
    sol = solvers.qp(P_,q_,G_,h_)

    # Get weight and b
    train_w = np.array(sol['x'])
    train_w = train_w[0:feature_num]
#     f = np.dot(train_x,train_w)
#     f[f < 0] = -1
#     f[f > 0] =  1 

#     #print training accurary
#     print ("Accuracy: "+str(100*np.count_nonzero(train_y*f+1)/row_training))
    return train_w
    #print Testing accurary

    # np.count_nonzero(train_y*f+1)
    
def testing(filename,w):
    data = load_data(filename)
    x = data[0]
    y = data[1]
    row_training = y.shape[0]
    f = np.dot(x,w)
    f[f < 0] = -1
    f[f > 0] =  1 
    accuracy = 100*np.count_nonzero(y*f+1)/row_training
    return accuracy
	
# The results
for i in range(0,9):
    c = pow(10,i)
    print ("Value of i: "+str(i))
    w = SVM_primal("wdbc_train.data",c)
    print("Training Accuracy:")
    print(testing("wdbc_train.data",w))
    print("Testing Accuracy:")
    print(testing("wdbc_test.data",w))
    print("Valid Accuracy:")
    print(testing("wdbc_valid.data",w))