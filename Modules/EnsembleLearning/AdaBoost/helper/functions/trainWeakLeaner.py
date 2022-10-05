from helper.imports.packageImports import np
from Modules.DecisionTree.helper.imports.functionImports import getAttrVals, buildTree

#trainWeakLeaner
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   weakLearner     :   Name of weak learner
#                                       Choices: ["decisionTree"]
#                   conditions      :   Dictionary of conditions to satisfy while training 
#                                       weak learner
#Output         :
#                   _               :   Trained weak learner
#What it does   :
#                   This function is used to perform train a weak learner
def trainWeakLeaner(X, weights, Y, weakLearner, conditions):
    if weakLearner == "decisionTree":
        attrVals = getAttrVals(X)
        attrsRem = np.arange(X.shape[1])
        labelType = "str"
        metric = "e"
        maxDepth = conditions["maxDepth"]
        root = buildTree(X, weights, Y, attrVals, 0, attrsRem, labelType, metric, maxDepth)
        return root 
    else: 
        return None