from Modules.DecisionTree.helper.imports.packageImports import np
#convertNumericalToCategorical
#Input          :
#                   X                 : Numpy matrix of data instances
#                   numericalAttrs    : List of indices of numerical attributes in X
#                   attrVals          : Dictionary of list of unique values taken by attributes 
#                                       in X indexed by the attribute index in X
#                   modify            : Boolean variable indicating whether attrVals needs to be 
#                                       updated   
#                                       Default: True
#Output         :
#                   X                 : Numpy matrix of preprocessed data instances
#                   attrVals          : Dictionary of list of unique values taken by attributes 
#                                       in X indexed by the attribute index in X
#What it does   :
#                   This function is used to convert numerical attributes into a boolean categorical
#                   attribute indicating whether the value of that attribute for a given data instance
#                   is greater than the median of all values for that attribute in X or not
def convertNumericalToCategorical(X, numericalAttrs, attrVals, modify=True):
    newCategoricalVals = []
    for n in numericalAttrs:
        median = np.median(X[:,n].astype(np.float64))
        newCategoricalVals.append((X[:,n].astype(np.float64)>median).astype(str))
        if modify:
            attrVals[n] = np.unique(newCategoricalVals[-1]).tolist()
    X[:,numericalAttrs] = np.array(newCategoricalVals).T
    return X, attrVals