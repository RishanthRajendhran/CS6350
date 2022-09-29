from helper.imports.functionImports import makePredictions, getError
#buildTree
#Input          :
#                   root    :   Root Node of the decision tree
#                   testX   :   Numpy matrix of test data instances
#Output         :
#                   _       :   Error on predictions made by the decision tree
#What it does   :
#                   This function is used to test a decision tree
def testTree(root, testX, testY):
    preds = makePredictions(root, testX)
    return getError(testY,preds)