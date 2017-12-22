import math
import pandas as pd
import numpy as np
from anytree import Node, RenderTree

# TRAINING
def Entropy(df):
    raw_data = df.groupby('result').size()
    sum = np.sum(raw_data)
    entropy = 0
    for element in raw_data:
        p = (element / sum)
        entropy = entropy - p * math.log(p,2)
    return entropy


# Pick the best attribute based on Information Gain
def attr_pick(S):
#     S_attr = S.drop('PlayTennis',axis=1) #get rid of result, just for Playtennis example
    S_attr = S.drop('result',axis=1) #get rid of result
    result = {}
    for Attr in list(S_attr):
        gain =  Entropy(S)
        p_list = S[Attr].value_counts()    
        for element in S[Attr].unique():
            p = p_list[element] / sum(p_list)
            entropy = Entropy(S[S[Attr] == element])  
            gain = gain - p*entropy
        result[Attr] = gain
        print("Gain of "+Attr+": "+str(gain))
    return max(result, key=result.get) #return name of the best Attribute 

# Check elements of an Attribute
def attr_elem_unicheck(Attr):
#     Attr - Result
#     freq = Attr['PlayTennis'].value_counts().to_dict()
    freq = Attr['result'].value_counts().to_dict()
    dominant = max(freq,key = freq.get)
    result = [dominant,len(freq)]
    return result

# d_tree(root,S,example)        
#  if uniform
#    if all +
#     create a + node
#     return nothing
#    if all -
#     create a - node
#     return nothing
#  else if (there is no any attribute)
#     create a node with majorite vote (+/-)
#     return nothing
#  else
#     Pick best attribute
#     next_node = create a node with the best attribute
#     For each label/categorize in the best attribute:
#       d_tree (next_node,new_s,label)

def d_tree(root,S,example_name):
    uniform_check = attr_elem_unicheck(S)
    if (uniform_check[1] == 1):
        if (uniform_check[0] == 'e'):
            Node(example_name + '/'+'e',parent = root)
            print("Yes")
            return
        else:
            Node(example_name + '/'+'p',parent = root)
            print("No")
            return
    elif (len(S.columns) == 2 & uniform_check[1] > 1): #Take the major vote
        Node(example_name + '/' + uniform_check[0],parent = root)
        print(uniform_check[0] + "- major vote")
        return
    else:        
        best_attr = attr_pick(S)
#         print(best_attr)
        next = Node(example_name +"/"+ best_attr,parent = root)
        for element in S[best_attr].unique():
            new_S = S[S[best_attr] == element] #Outlook = Sunny
            new_S = new_S.drop(best_attr,axis=1)
            print("new_S size: "+ str(new_S.columns))
            d_tree(next,new_S,element)

#Example
# root = Node('Start')
# d_tree(root,df,'')  
# -----------------------------------------------------------------------------
# TESTING
def get_name(node):
    return node.name.split('/')

def find_children(node,result):
    for child in node.children:
        if get_name(child)[0] == result:
            return child
        
def traverse_tree(example,root):
    if (root.is_leaf):
        result = get_name(root)[1]
#         print(result)
        return result
#         print("WTF")
    else:        
        node_name = example[get_name(root)[1]] #Get name of next node
#         print(node_name)
        child = find_children(root,node_name)
#         print(child)
        return traverse_tree(example,child)

def test(test_data,root):
    accuracy = 0
    total = test_data.shape[0]
    for index, row in test_data.iterrows():
        predict = traverse_tree(row,root.children[0])
        actual = row['result']
#         print ("pre: "+predict+" actual: "+actual)
        if (predict == actual):
            accuracy = accuracy + 1
    return("Accuracy = "+str(accuracy/total))
	
#Result
#Read Input
import pandas as pd
train_data = pd.read_table('./mush_train.data', delimiter =',',names = ('result', 'cap-shape','cap-surface','cap-color',
                                                                 'bruises?','odor','gill-attachment','gill-spacing',
                                                                 'gill-size','gill-color','stalk-shape','stalk-root',
                                                                 'stalk-surface-above-ring','stalk-surface-below-ring',
                                                                 'stalk-color-above-ring','stalk-color-below-ring',
                                                                 'veil-type','veil-color','ring-number','ring-type',
                                                                 'spore-print-color','population','habitat'))

# Train data
root = Node('Start')
d_tree(root,train_data,'') 

# Run test on training data and report
print("Test on training data")
test(train_data,root)

#Print Tree
for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))
	1

# load testing file
test_data = pd.read_table('./mush_test.data', delimiter =',',names = ('result', 'cap-shape','cap-surface','cap-color',
                                                                 'bruises?','odor','gill-attachment','gill-spacing',
                                                                 'gill-size','gill-color','stalk-shape','stalk-root',
                                                                 'stalk-surface-above-ring','stalk-surface-below-ring',
                                                                 'stalk-color-above-ring','stalk-color-below-ring',
                                                                 'veil-type','veil-color','ring-number','ring-type',
                                                                 'spore-print-color','population','habitat'))
# Run Test
print(test(test_data,root))