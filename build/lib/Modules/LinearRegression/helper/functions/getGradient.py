from Modules.DecisionTree.helper.imports.packageImports import np

#getGradient
#Input          :
#                   X           :   Numpy matrix of data instances
#                   weights     :   Numpy array of weights of data instances in X
#                   Y           :   Numpy array of target labels of instances in X
#                   w           :   Numpy array of initial weights of linear regressor
#                   loss        :   Type of loss function to use
#                                   Default: lms
#Output         :
#                   _           :   Numpy array of gradient of weights of 
#                                   linear regressor after GD
#What it does   :
#                   This function is used to calculate gradient of weights
def getGradient(X, weights, Y, w, loss="lms"):
    if loss=="lms":
        return np.sum(np.dot((-(Y-np.dot(X,w))),X),axis=0)
    return None
