import numpy as np 
import csv
import sys

if sys.argv[1] == "car":
    trainPath = "./car/train.csv" 
    testPath = "./car/test.csv"
    attrNames = ["buying","maint","doors","persons","lug_boot","safety","label"]
    labelNames = ["No", "Yes"]
    attrVals = {
        0:   ["vhigh", "high", "med", "low"],
        1:    ["vhigh", "high", "med", "low"],
        2:    ["2", "3", "4", "5more"],
        3:  ["2", "4", "more"],
        4: ["small", "med", "big"],
        5:   ["low", "med", "high"],
    }
    labelType = str
    maxTreeDepths = np.arange(6)
elif sys.argv[1] == "bank":
    trainPath = "./bank/train.csv" 
    testPath = "./bank/test.csv"
    attrNames = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]
    labelNames = ["No", "Yes"]
    categoricalAttrs = [1,2,3,4,6,7,8,10,15]
    numericalAttrs = [0,5,9,11,12,13,14]
    attrVals = {
        1:   ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
        2:    ["married","divorced","single"],
        3:    ["unknown","secondary","primary","tertiary"],
        4:  ["yes","no"],
        6: ["yes","no"],
        7:   ["yes","no"],
        8:   ["unknown","telephone","cellular"],
        10:    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        15:    ["unknown","other","failure","success"],
    }
    labelType = str
    maxTreeDepths = np.arange(16)

#getEntropy
#Input          :
#                   ns : A numerical type list of counts of all class labels
#Output         :
#                   Entropy rounded to 4 decimal places
# What it does  :
#                   This function computes the entropy of a given data set given
#                   the numerical counts of all class labels in that data set
def getEntropy(ns):
    if len(ns)==0:
        return 0
    ent = 0
    N = np.sum(ns)
    for i in range(len(ns)):
        p = ns[i]/N
        if p:
            ent -= p*np.log2(p)
    return np.round(ent,4)

#getMajorityError
#Input          :
#                   ns : A numerical type list of counts of all class labels
#Output         :
#                   Majority Error rounded to 4 decimal places
# What it does  :
#                   This function computes the majority error of a given data set given
#                   the numerical counts of all class labels in that data set
def getMajorityError(ns):
    if len(ns)==0:
        return 0
    N = np.sum(ns)
    me = (N-max(ns))/N
    return np.round(me,4)

#getGiniIndex
#Input          :
#                   ns : A numerical type list of counts of all class labels
#Output         :
#                   Gini Index rounded to 4 decimal places
# What it does  :
#                   This function computes the gini index of a given data set given
#                   the numerical counts of all class labels in that data set
def getGiniIndex(ns):
    if len(ns)==0:
        return 0
    N = np.sum(ns)
    gini = 1 - np.sum((np.array(ns)/N)**2)
    return np.round(gini,4)

metricsMap = {
    "e": getEntropy,
    "m": getMajorityError,
    "g": getGiniIndex,
}

#getGain
#Input          :
#                   ns          : A numerical type list of counts of all class labels
#                   numClasses  : Number of class labels
#                   metric      : Metric to calculate gain with 
#Output         :
#                   Gain rounded to 4 decimal places
# What it does  :
#                   This function computes the gain of a split of a given data set over an attribute given
#                   the probabilities of all class labels in the original and split data sets
def getGain(ns, numClasses=2, metric="e"):  
    if len(ns)%numClasses:
        print("Unexpected size for ns")
        exit(0)
    vals = [metricsMap[metric](np.array(ns)[i*numClasses:i*numClasses+numClasses]) for i in range(len(ns)//numClasses)]
    gain = vals[0]
    for i in range(1,len(vals)):
        gain -= (np.sum(np.array(ns)[i*numClasses:i*numClasses+numClasses])/np.sum(np.array(ns[0:numClasses])))*vals[i]
    return np.round(gain,4)

class Node:
    def __init__(self,attr,depth,isLeafNode=False,label=None):
        self.attr = attr 
        self.children = []
        self.childrenAttrVal = []
        self.depth = depth
        self.isLeafNode = isLeafNode
        self.label = label
    def addChild(self, child, attrVal):
        self.children.append(child)
        self.childrenAttrVal.append(attrVal)
    def printTree(self,n=0):
        st = "\t"*n
        print(f"{st}Attr {self.attr}, Depth {self.depth}")
        if self.isLeafNode:
            print(f"{st}Label: {self.label}")   
        for ch in range(len(self.children)):
            print(f"{st}\tAttrVal = {self.childrenAttrVal[ch]}")
            self.children[ch].printTree(n+2)
    def predict(self, x):
        if self.isLeafNode:
            return self.label
        return self.children[self.childrenAttrVal.index(x[self.attr])].predict(x)


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

def buildTree(X, Y, attrVals, depth, attrsRem = [], metric = "e", maxDepth=np.inf):
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
                child = buildTree(bestXValYs[attrVal][:,:-1],bestXValYs[attrVal][:,-1].astype(labelType), attrVals, depth+1, attrsRem, metric, maxDepth)
            if child == None:
                vals, counts = np.unique(Y,return_counts=True)
                maxLabel = vals[np.argmax(counts)]
                child = Node(None, depth+1,True,maxLabel)
                # child = Node(None, depth+1,True,np.argmax(np.bincount(Y)))
            curNode.addChild(child,attrVal)
    return curNode

def getAccuracy(Y, Preds):
    if len(Y) != len(Preds):
        print("Y and Preds are not of same size!")
        return 0
    return np.sum(np.array(Preds)==np.array(Y))/len(Y)

def getError(Y, Preds):
    return np.round(1-getAccuracy(Y,Preds),4)

def makePredictions(root, X):
    preds = []
    for x in X:
        preds.append(root.predict(x))
    return preds

def testTree(root, testX, testY):
    preds = makePredictions(root, testX)
    return getError(testY,preds)

def getAttrVals(X,locs=None):
    if locs == None:
        locs = np.arange(X.shape[1])
    vals = {}
    for loc in locs:
        vals[loc] = np.unique(X[:,loc])
    return vals

def convertNumericalToCategorical(X, numericalAttrs, attrVals, modify=True):
    newCategoricalVals = []
    for n in numericalAttrs:
        median = np.median(X[:,n].astype(np.float64))
        newCategoricalVals.append((X[:,n].astype(np.float64)>median).astype(str))
        if modify:
            attrVals[n] = np.unique(newCategoricalVals[-1]).tolist()
    X[:,numericalAttrs] = np.array(newCategoricalVals).T
    return X, attrVals

def handleMissingAttrValues(X, categoricalAttrs, attrVals, modify=True):
    for attr in categoricalAttrs:
        if "unknown" in X[:,attr]:
            vals, counts = np.unique(X[:,attr],return_counts=True)
            vals = vals.tolist()
            counts = counts.tolist()
            if "unknown" in vals:
                ind = vals.index("unknown")
                vals.remove("unknown")
                counts.remove(counts[ind])
            X[np.where(X[:,attr] == "unknown")[0],attr] = vals[np.argmax(counts)]
            if modify:
                attrVals[attr].remove("unknown")
    return X, attrVals

with open(trainPath, newline="") as csvFile:
    train = np.array(list(csv.reader(csvFile, delimiter=",")))[1:,:]

with open(testPath, newline="") as csvFile:
    test = np.array(list(csv.reader(csvFile, delimiter=",")))[1:,:]

X = train[:,:-1].copy()
Y = train[:,-1].copy()
testX = test[:,:-1].copy()
testY = test[:,-1].copy()

if sys.argv[1] == "bank":
    X, attrVals = convertNumericalToCategorical(X,numericalAttrs,attrVals)
    X, attrVals = handleMissingAttrValues(X,categoricalAttrs,attrVals)
    testX, _ = convertNumericalToCategorical(testX,numericalAttrs,attrVals,False)
    testX, _ = handleMissingAttrValues(testX,categoricalAttrs,attrVals,False)

# X = np.array([['sunny', 'hot', "high", 'weak'],
#         ['sunny', 'hot', "high", 'strong'],
#         ['overcast', "hot", 'high', 'weak'],
#         ['rainy', "medium", 'high', 'weak'],
#         ['rainy', "cool", 'normal', 'weak'],
#         ['rainy', "cool", 'normal', 'strong'],
#         ['overcast', "cool", 'normal', 'strong'],
#         ['sunny', "medium", 'high', 'weak'],
#         ['sunny', "cool", 'normal', 'weak'],
#         ['rainy', "medium", 'normal', 'weak'],
#         ['sunny', "medium", 'normal', 'strong'],
#         ['overcast', "medium", 'high', 'strong'],
#         ['overcast', "hot", 'normal', 'weak'],
#         ['rainy', "medium", 'high', 'strong']])
# Y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
if attrVals == None:
    attrVals = getAttrVals(X)
    
trainPredErrors = []
testPredErrors = []
for maxDepth in maxTreeDepths:
    trainPredError = []
    testPredError = []
    for metric in metricsMap.keys():
        root = buildTree(X,Y,attrVals,1,np.arange(X.shape[1]),metric, maxDepth)
        trainPredError.append(testTree(root, X, Y))
        testPredError.append(testTree(root, testX, testY))
    trainPredErrors.append(trainPredError)
    testPredErrors.append(testPredError)
print("Average Prediction Errors:")
print('{:<15}'.format("metric ->"),end="")
for metric in metricsMap.keys():
    print('{:<15}'.format(metric),end="")
print("\n")
print('{:<15}'.format("maxDepth |"),end="\n")
print('{:>10}'.format("V"),end="\n")
print("Train Data set:")
for maxDepth in range(len(trainPredErrors)):
    print('{:<15}'.format(maxDepth+1),end="")
    for metric in range(len(trainPredErrors[maxDepth])):
        print('{:<15}'.format(str(trainPredErrors[maxDepth][metric])),end="")
    print("\n")
print("Test Data set:")
for maxDepth in range(len(testPredErrors)):
    print('{:<15}'.format(maxDepth+1),end="")
    for metric in range(len(testPredErrors[maxDepth])):
        print('{:<15}'.format(str(testPredErrors[maxDepth][metric])),end="")
    print("\n")



    