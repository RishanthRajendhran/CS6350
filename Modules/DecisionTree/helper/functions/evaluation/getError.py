from Modules.DecisionTree.helper.imports.packageImports import np
from Modules.DecisionTree.helper.functions.evaluation.getAccuracy import getAccuracy
#getError
#Input          :
#                   Y        : Numpy matrix of target labels of instances in X
#                   Preds    : Numpy array of predictions 
#                   weights  : Weights of test data instances
#                              Default: All instances have a(n equal) weight of 1
#Output         :
#                   _        : Error on predictions
#What it does   :
#                   This function is used to compute weighted error rounded to 4 
#                   decimal places
def getError(Y, Preds, weights=[]):
    return 1-getAccuracy(Y,Preds,weights)