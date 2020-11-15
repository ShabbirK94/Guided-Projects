from sklearn import datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(0)

#Generating a dataset from sklearn.
features, labels = datasets.make_moons(100, noise=0.1)
plt.figure(figsize= (10, 7))
plt.scatter(features[:, 0], features[:, 1], c= labels, cmap=plt.cm.winter)

labels= labels.reshape(100, 1)

#plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))

wh= np.random.rand(len(features[0]), 4)
# wh--> weights of hidden layers 4 is the number of nodes in each hidden layer
wo = np.random.rand(4, 1)
# wo--> weights of output layer

#learning rate initialization
lr= 0.5

for epoch in range(20000):
    ##Feedforward Propagation
    zh= np.dot(features, wh) ##Dot product of feature set and hidden weights
    ah= sigmoid(zh)          ##Sigmoid of Dot Product of feature_set and hidden weights

    zo= np.dot(ah, wo)       ##Dot product of output weights and the sigmoid of Dot product of feature_set and hidden weights
    ao= sigmoid(zo)          ##Output equals Sigmoid of Dot Product of output weights and the sigmoid of
                             ##Dot product of feature_set and hidden weights

    ##Back Propagation
    #Phase 1
    error_out = ((1/2) * (np.power((ao - labels), 2))) #Loss
    print(error_out.sum())
    dcost_dao = ao - labels ##Derivative cost
    dao_dzo = sigmoid_der(zo) ## Sigmoidal derivative of output
    dzo_dwo= ah               ## Derivative of sigmoid output of dot-product of sigmoid of hidden layer weights

    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo) #Derivative cost of output weights equals dot product of
                                                      ##derivative of sigmoid output of dot-product of sigmoid of
                                                      ##hidden layer weights (transposed) and the multiplication of
                                                      ##Derivative cost and sigmoidal derivative of output
    

    #Phase 2

    dcost_dzo = dcost_dao + dao_dzo ##Final Derivative Cost of Final Output equals derivative cost of output multiplied by
                                    ##the sigmoidal derivative of output
    dzo_dah = wo                    ##Output Weights
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T) ##Derivative cost of dot product of sigmoid of hidden layer weights equals
                                             ## dot product of dcost_dzo and transposition of output weights

    dah_dzh = sigmoid_der(zh)
    dzh_dwh = features
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)


    #Update weights
    wh -= lr *dcost_wh
    wo -= lr *dcost_wo





    

    
