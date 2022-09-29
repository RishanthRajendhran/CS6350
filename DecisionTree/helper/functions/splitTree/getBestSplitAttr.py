from helper.imports.packageImports import np
from helper.imports.functionImports import getGain
#getBestSplitAttr
#Input          :
#                   X               :   Numpy matrix of data instances
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
def getBestSplitAttr(X,Y, attrsRem, metric="e"):
    X = np.array(X)
    Y = np.array(Y)
    bestAttr = None
    bestAttrVals = None
    bestXValYs = None
    maxGain = 0
    classes = np.unique(Y)
    counts = []
    XY = np.concatenate((X, Y.reshape((Y.shape[0],1))),axis=1)
    for clas in classes:
        counts.append(len(np.where(XY[:,-1]==str(clas))[0]))
    for attr in range(len(attrsRem)):
        newCounts = counts.copy()
        vals = np.unique(X[:,attrsRem[attr]])
        xValys = {}
        for val in vals:
            xVal = X[np.where(X[:,attrsRem[attr]] == str(val))]
            yVal = Y[np.where(X[:,attrsRem[attr]] == str(val))]
            xValy = np.concatenate((xVal, yVal.reshape((yVal.shape[0],1))),axis=1)
            xValys[val] = (xValy)
            for clas in classes:
                newCounts.append(len(np.where(xValy[:,-1]==str(clas))[0]))
        gain = getGain(newCounts,len(classes),metric)
        if gain > maxGain or bestAttr == None:
            maxGain = gain 
            bestAttr = attrsRem[attr]
            bestAttrVals = vals
            bestXValYs = xValys
    return (bestAttr, bestAttrVals, bestXValYs)