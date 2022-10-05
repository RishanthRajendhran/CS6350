from helper.imports.packageImports import np
from Modules.DecisionTree.helper.imports.functionImports import testTree, getError, getAccuracy

#makePredictions
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   h               :   Trained weak learner
#                   weakLearner     :   Name of weak learner h
#                                       Choices: ["decisionTree"]
#Output         :
#                   preds           :   Predictions made by weak learner on X
#                   _               :   Prediction error
#                   _               :   Prediction accuracy
#What it does   :
#                   This function is used to make predictions using a weak learner
def makePredictions(X, weights, Y, h, weakLearner):
    if weakLearner == "decisionTree":
        preds, err = testTree(h, X, Y, weights)
        return preds, getError(Y, preds, weights), getAccuracy(Y, preds, weights)
    else: 
        return None, None, None