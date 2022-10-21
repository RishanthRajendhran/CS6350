from Modules.DecisionTree.helper.functions.treeOperations.makePredictions import makePredictions
from Modules.DecisionTree.helper.functions.evaluation.getError import getError
#buildTree
#Input          :
#                   root    :   Root Node of the decision tree
#                   testX   :   Numpy matrix of test data instances
#                   testY   :   Numpy matric of test target labels
#                   weights :   Numpy vector of weights of test dataa instances
#                               Default: All data instances weighted uniformly
#Output         :
#                   preds   :   Predictions made by the tree on testX
#                   _       :   Error on predictions made by the decision tree
#What it does   :
#                   This function is used to test a decision tree
def testTree(root, testX, testY, weights=[]):
    preds = makePredictions(root, testX)
    return preds, getError(testY, preds, weights)