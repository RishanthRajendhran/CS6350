from helper.imports.packageImports import np
from helper.imports.functionImports import getBestSplitAttr
from helper.imports.classImports import Node
#buildTree
#Input          :
#                   X               :   Numpy matrix of data instances
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
def buildTree(X, Y, attrVals, depth, attrsRem = [], labelType="str", metric = "e", maxDepth=np.inf):
    if len(X) == 0 or len(attrsRem) == 0:
        return None
    bestAttr, bestAttrVals, bestXValYs = getBestSplitAttr(X,Y,attrsRem,metric)
    attrsRem = np.delete(attrsRem, attrsRem.tolist().index(bestAttr))
    if not (maxDepth-depth):
        vals, counts = np.unique(Y,return_counts=True)
        maxLabel = vals[np.argmax(counts)]
        return Node(None, depth,True,maxLabel)
        # return Node(None, depth,True,np.argmax(np.bincount(Y)))
    if len(np.unique(Y)) == 1:
        curNode = Node(bestAttr, depth, True, Y[0])
    else: 
        curNode = Node(bestAttr, depth)
        for attrVal in attrVals[bestAttr]:
            child = None
            if attrVal in bestXValYs.keys(): 
                child = buildTree(bestXValYs[attrVal][:,:-1],bestXValYs[attrVal][:,-1].astype(labelType), attrVals, depth+1, attrsRem, labelType, metric, maxDepth)
            if child == None:
                vals, counts = np.unique(Y,return_counts=True)
                maxLabel = vals[np.argmax(counts)]
                child = Node(None, depth+1,True,maxLabel)
                # child = Node(None, depth+1,True,np.argmax(np.bincount(Y)))
            curNode.addChild(child,attrVal)
    return curNode