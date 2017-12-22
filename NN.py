# import library
import numpy as np
from random import *

def relu(z):
    """
    Compute the RELU of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- RELU(z)
    """
    s = np.maximum(0,z)    
    return s

def softplus(z):
    """
    Compute the RELU of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- RELU(z)
    """
    s = np.log(1 + np.exp(z))
    return s

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###
    
    return s
def relu_de(z):
    s = 1 * z
    s[s>=0] = 1
    s[s<0] = 0
    return s
 
def generate_input(m):
    X = np.random.randint(2, size=(10,m))
    Y = np.zeros((1,m))
    for i in range(0,m):
        num_ones = len(np.nonzero(X[:,i])[0])
        if (num_ones % 3 == 0):
            Y[0,i] = 1
    return [X,Y]
	
def intialize(X,hidden_layer):

    num_attr = X.shape[0]
    layer = 1 * hidden_layer
    layer.insert(0,num_attr)
    layer.append(1)
    num_layer = len(layer)-1
    w_list = []
    b = [0] * (num_layer)
#     b = [0] * num_layer
    print("num of layer: ",num_layer)
    for i in range(0,num_layer):
#         w = np.zeros((layer[i],layer[i+1]))
        w = 0.5*np.ones((layer[i],layer[i+1]))
        print(w.shape)
        w_list.append(w)
    return [w_list,b]
	
def propagate(w,b, X, Y,hidden_layer,learning_rate,activate):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
   
    w -- weights, a list contains numpy array of weight for each layer, w[i] -> weight for layer i (there is no layer 0)
    hidden_layer -- array defines num of neurons in each hidden layer e.g.  [2,3] -> 2 neurons in first layer, 3 in second one
    X -- Input 
    Y -- Output
    b -- bias
    
    a -- activation a = activation(z)
    z -- linear combination
    
    Size:
    n: num of atrributes
    m: num of example
    w[i] -> ( N_neurons previous layer[i - 1], N_neurons at current layer [i])
    w[0] -> (10,3)
    w[1] -> (3,2)
    w[2] -> (2,1)
    
    b -> (1 , N_layer + 1 )
    
    X -> (n, m)
    Y -> (1, m)
    
    a[i] -> (N_neurons at current layer[i] , m)
    z[i] = w[i].T * a[i] (size = a[i])
    (3,10) * (10,m)
    
    dw -> size = w
    dz -> size = z
    db -> size = 1
    Return:

    """
    def activate_f(z):
        if (activate == 'softplus'):
            return softplus(z)
        if (activate =='relu'):
            return relu(z)
        if (activate == 'perceptron'):
            z_ = 1*z
            z_[z_>0] = 1
            z_[z_<=0] = 0
            return z_
    def deri_f(z):
        if (activate == 'softplus'):
            return sigmoid(-z)
        if (activate == 'relu'):
            z_ = 1 * z
            z_[z_>0] = 1
            z_[z_<=0] = 0
            return z_
        if (activate == 'perceptron'):
            return 1
        
    # Initialize parameter
    m = X.shape[1]
    layer = 1 * hidden_layer
    layer.append(1)
    layer_num = len(layer)-1
    a_list = [] 
    a_list.append(X)    
    z_list = [] 
#     print("layer num: ",layer_num)
    
    # Forward 
#     print("Forward Step")
    for i in range(0,layer_num +1):
#         print('i: ',i)
        z = (np.dot(w[i].T,a_list[i]) + b[i])  

        z_list.append(z)      
#         a = relu(z)
#         a = softplus(z)
        a = activate_f(z)
        a_list.append(a)
    
#         print('a at layer ',(i+1),' is: ',a_list[i])
#         print('w at layer ',(i+1),' is: ',w[i])
#         print('b at layer ',(i+1),' is: ',b[i])    
#         print('z at layer ',(i+1),' is: ',z)
    
    A = a_list[layer_num+1]
#     print('A: ',A)
#     cost = - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m  # compute cost
    cost =  np.sum(np.square(Y - A)) / (2*m)
#     print('cost: ',cost)
    
    # Back propagation   
#     print("Backward step")
    dw_list = []    
    dz_list = [None] * (layer_num+1)
    z = z_list[layer_num]
#     dz = np.sum(A - Y) * relu_de(z)
    dz = np.sum(A - Y) * deri_f(z) / m
#     print(A)
#     print(Y)
#     print("A-Y = ",A-Y)
#     print("da/dz =",sigmoid(-z))
#     dz = (-Y/a_list[layer_num] + (1-Y) / (1 - a_list[layer_num])) / np.exp(Y)
    
# #     dz = (-Y/a_list[layer_num] + (1-Y) / (1 - a_list[layer_num])) * Y
    dz_list[layer_num] = dz
    for i in range(layer_num,-1,-1):
        dw = np.dot(a_list[i],dz_list[i].T) / m
#         # dw[0] = X.T * dz[0] - (n,m) * (3,m).T = (10,3)
        db = np.sum(dz_list[i]) / m 
    
#         print("dz at layer ",(i+1),' is: ',dz_list[i])
#         print("dw at layer ",(i+1),' is: ',dw)
#         print("db at layer ",(i+1),' is: ',db)
        w[i] = w[i] - learning_rate * dw
        b[i] = b[i] - learning_rate * db
        if (i != 0):
#             dz[i-1] = np.dot(dz[i], w[i].T) / np.exp(a[i])
#             dz_list[i-1] = np.dot(w[i],dz_list[i]) * relu_de(a_list[i])
            dz_list[i-1] = np.dot(w[i],dz_list[i]) * deri_f(a_list[i]) / m
#     print('a list: ',a_list)
#     print('dz list: ',dz_list)
    return [cost,w,b]
	
# Generate input
[X,Y] = generate_input(100)

hidden_layer = []
[w,b] = intialize(X,hidden_layer)

[w,b] = intialize(X,hidden_layer)
learning_rate = 0.1
iter_num = 10000
w_1 = 1*w
b_1 = 1*b
for i in range(0,iter_num):
    activate = "relu"
    [cost,w_1,b_1] = propagate(w_1,b_1, X, Y,hidden_layer,learning_rate,activate)
    print (cost)
print("Cost: ",cost)