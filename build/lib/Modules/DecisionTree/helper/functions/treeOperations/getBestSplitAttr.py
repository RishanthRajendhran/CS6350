from Modules.DecisionTree.helper.imports.packageImports import np
from Modules.DecisionTree.helper.functions.gain.getGain import getGain
#getBestSplitAttr
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy matrix of target labels of instances in X
#                   attrsRem        :   List of remaining attributes on which splitting can be done
#                   metric          :   Metric to be used for computing gain
#                                       Default: "e" (for "e"ntropy as defined in miscConfig.py)
#Output         :
#                   bestAttr        :   Best attribute to perform split on
#                   bestAttrVals    :   List of unique values taken by the attribute
#                   bestXValYs      :   Dictionary of list of data samples in X with a given value
#                                       of the attribute indexed by the value
#What it does   :
#                   This function is used to find the best attribute to split the given data matrix
#                   on given the set of attributes and the metric to use for computing gain
#Assumption     :
#                   It is assumed that all values that all attributes can take can be inferred from
#                   the input X
def getBestSplitAttr(X, weights, Y, attrsRem, metric="e"):
    X = np.array(X)
    Y = np.array(Y)

    bestAttr = None
    bestAttrVals = None
    bestXValYs = None
    maxGain = 0
    classes = np.unique(Y)
    counts = []

    for clas in classes:
        counts.append(np.sum(weights[np.where(Y==str(clas))[0]].astype(np.float64)))

    for attr in range(len(attrsRem)):
        newCounts = []
        newCounts.append(counts.copy())
        vals = np.unique(X[:,attrsRem[attr]])
        xValys = {}
        for val in vals:
            reqInds = np.where(X[:,attrsRem[attr]] == str(val))[0]
            xVal = X[reqInds]
            yVal = Y[reqInds]
            wVal = weights[reqInds]
            xValy = np.concatenate((wVal.reshape((wVal.shape[0],1)), xVal), axis=1)
            xValy = np.concatenate((xValy, yVal.reshape((yVal.shape[0],1))),axis=1)
            xValys[val] = (xValy)
            curCount = [0]*len(classes)
            for clas in range(len(classes)):
                curCount[clas] = np.sum(wVal[np.where(xValy[:,-1]==str(classes[clas]))[0]].astype(np.float64))
            newCounts.append(curCount)
        gain = getGain(newCounts,len(classes),metric)
        if gain > maxGain or bestAttr == None:
            maxGain = gain 
            bestAttr = attrsRem[attr]
            bestAttrVals = vals
            bestXValYs = xValys
    return (bestAttr, bestAttrVals, bestXValYs)