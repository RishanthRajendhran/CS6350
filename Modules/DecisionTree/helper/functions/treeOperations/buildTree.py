from Modules.DecisionTree.helper.imports.packageImports import np
from Modules.DecisionTree.helper.imports.classImports import Node
from Modules.DecisionTree.helper.functions.treeOperations.getBestSplitAttr import getBestSplitAttr
#buildTree
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   attrVals        :   Dictionary of list of unique values taken by attributes 
#                                           in X indexed by the attribute index in X
#                   depth           :   Current depth of the tree
#                   attrsRem        :   List of remaining attributes on which splitting can be done
#                   labelType       :   Data type of target labels
#                   metric          :   Metric to be used for computing gain
#                                       Default: "e" (for "e"ntropy as defined in miscConfig.py)
#                   maxDepth        :   Maximum allowed depth of the tree  
#                                       Default: np.inf
#Output         :
#                   curNode         :   Node at the given depth
#What it does   :
#                   This function is used to build a decision tree recursively
def buildTree(X, weights, Y, attrVals, depth, attrsRem = [], labelType="str", metric = "e", maxDepth=np.inf):
    if len(X) == 0 or len(attrsRem) == 0:
        return None
    if not (maxDepth-depth):
        vals= np.unique(Y)
        counts = [0]*len(vals)
        for clas in range(len(vals)):
            reqInds = np.where(Y==vals[clas])[0]
            counts[clas] = np.sum(weights[reqInds].astype(np.float64))
        maxLabel = vals[np.argmax(counts)]
        return Node(None, depth,True,maxLabel)
    bestAttr, _, bestXValYs = getBestSplitAttr(X, weights, Y,attrsRem,metric)
    attrsRem = np.delete(attrsRem, attrsRem.tolist().index(bestAttr))
    if len(np.unique(Y)) == 1:
        curNode = Node(bestAttr, depth, True, Y[0])
    else: 
        curNode = Node(bestAttr, depth)
        for attrVal in attrVals[bestAttr]:
            child = None
            if attrVal in bestXValYs.keys(): 
                bestX = bestXValYs[attrVal][:,1:-1]
                bestWeights = bestXValYs[attrVal][:,0]
                bestY = bestXValYs[attrVal][:,-1].astype(labelType)
                child = buildTree(bestX, bestWeights, bestY, attrVals, depth+1, attrsRem, labelType, metric, maxDepth)
            if child == None:
                vals= np.unique(Y)
                counts = [0]*len(vals)
                for clas in range(len(vals)):
                    reqInds = np.where(Y==vals[clas])[0]
                    counts[clas] = np.sum(weights[reqInds].astype(np.float64))
                maxLabel = vals[np.argmax(counts)]
                child = Node(None, depth+1,True,maxLabel)
            curNode.addChild(child,attrVal)
    return curNode