from helper.imports.packageImports import *
from helper.imports.functionImports import *
from helper.imports.configImports import * 
from helper.imports.classImports import *

dataSetName = sys.argv[1]

train = readCSV(dataConfig.dataSets[dataSetName]["trainPath"], True)
test = readCSV(dataConfig.dataSets[dataSetName]["testPath"], True)

X = train[:,:-1].copy()
Y = train[:,-1].copy()
testX = test[:,:-1].copy()
testY = test[:,-1].copy()

if dataSetName == "bank":
    X, dataConfig.dataSets[dataSetName]["attrVals"] = convertNumericalToCategorical(X,dataConfig.dataSets[dataSetName]["numericalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
    X, dataConfig.dataSets[dataSetName]["attrVals"] = handleMissingAttrValues(X,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
    testX, _ = convertNumericalToCategorical(testX,dataConfig.dataSets[dataSetName]["numericalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"],False)
    testX, _ = handleMissingAttrValues(testX,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"],False)

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
if dataConfig.dataSets[dataSetName]["attrVals"] == None:
    dataConfig.dataSets[dataSetName]["attrVals"] = getAttrVals(X)
    
#Temporary
weights = np.ones((len(X),))
#Temporary

trainPredErrors = []
testPredErrors = []
for maxDepth in dataConfig.dataSets[dataSetName]["maxTreeDepths"]:
    trainPredError = []
    testPredError = []
    for metric in miscConfig.metricsMap.keys():
        root = buildTree(X,weights,Y,dataConfig.dataSets[dataSetName]["attrVals"],1,np.arange(X.shape[1]),dataConfig.dataSets[dataSetName]["labelType"],metric, maxDepth)
        trainPredError.append(testTree(root, X, Y))
        testPredError.append(testTree(root, testX, testY))
    trainPredErrors.append(trainPredError)
    testPredErrors.append(testPredError)
print("Average Prediction Errors:")
print('{:<15}'.format("metric ->"),end="")
for metric in miscConfig.metricsMap.keys():
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



    