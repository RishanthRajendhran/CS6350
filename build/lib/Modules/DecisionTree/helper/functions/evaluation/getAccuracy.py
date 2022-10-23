from Modules.DecisionTree.helper.imports.packageImports import np
#getAccuracy
#Input          :
#                   Y        : Numpy matrix of target labels of instances in X
#                   Preds    : Numpy array of predictions 
#                   weights  : Weights of test data instances
#                              Default: All instances have a(n equal) weight of 1
#Output         :
#                   _        : Accuracy on predictions
#What it does   :
#                   This function is used to compute weighted accuracy
def getAccuracy(Y, Preds, weights=[]):
    if len(weights) == 0:
        weights = np.ones((len(Y),))
        weights /= len(Y)
    if len(Y) != len(Preds):
        print("Y and Preds are not of same size!")
        return 0
    return np.sum((np.array(Preds)==np.array(Y))*weights.astype(float))/np.sum(weights)
    # return np.sum(weights[np.where(np.array(Preds)==np.array(Y))[0]].astype(float))