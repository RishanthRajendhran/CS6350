from Modules.DecisionTree.helper.imports.packageImports import *
from Modules.DecisionTree.helper.imports.functionImports import *
from Modules.DecisionTree.helper.imports.configImports import * 
from Modules.DecisionTree.helper.imports.classImports import *

parser = argparse.ArgumentParser()
parser.add_argument("-ds",
                    "--dataset",
                    help="Data set to use for building decision tree",
                    choices=["car", "bank", "creditCard"]
                )
parser.add_argument("-hmv",
                    "--handleMissingVals",
                    action="store_true",
                    help="Boolean flag indicating whether missing values need to be imputed"
                )

parser.add_argument("-md",
                    "--maxDepth",
                    nargs="+",
                    type=int,
                    help="Maximum allowed depth(s) for the decision tree; multiple values should be separated by a whitespace"
                )
args = parser.parse_args()
if args.dataset:
    dataSetName = args.dataset
else: 
    print(f"No dataset provided!")
    exit(0)

miscConfig.handleMissingVals = args.handleMissingVals

if args.maxDepth:
    dataConfig.dataSets[dataSetName]["maxTreeDepths"] = args.maxDepth

train = readCSV(dataConfig.dataSets[dataSetName]["trainPath"], True)
test = readCSV(dataConfig.dataSets[dataSetName]["testPath"], True)

X = train[:,:-1].copy()
Y = train[:,-1].copy()
testX = test[:,:-1].copy()
testY = test[:,-1].copy()

if dataConfig.dataSets[dataSetName]["attrVals"] == None:
    dataConfig.dataSets[dataSetName]["attrVals"] = getAttrVals(X)

X, dataConfig.dataSets[dataSetName]["attrVals"] = convertNumericalToCategorical(X,dataConfig.dataSets[dataSetName]["numericalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
if miscConfig.handleMissingVals:
    X, dataConfig.dataSets[dataSetName]["attrVals"] = handleMissingAttrValues(X,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
testX, _ = convertNumericalToCategorical(testX,dataConfig.dataSets[dataSetName]["numericalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"],False)
if miscConfig.handleMissingVals:
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
    
#Temporary
weights = np.ones((len(X),))
weights /= np.sum(weights)
#Temporary

trainPredErrors = []
testPredErrors = []
for maxDepth in dataConfig.dataSets[dataSetName]["maxTreeDepths"]:
    trainPredError = []
    testPredError = []
    for metric in miscConfig.metricsMap.keys():
        root = buildTree(X,weights,Y,dataConfig.dataSets[dataSetName]["attrVals"],1,np.arange(X.shape[1]),dataConfig.dataSets[dataSetName]["labelType"],metric, maxDepth)
        preds, err = testTree(root, X, Y)
        trainPredError.append(err)
        preds, err = testTree(root, testX, testY)
        testPredError.append(err)
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
for maxDepth in dataConfig.dataSets[dataSetName]["maxTreeDepths"]:
    print('{:<15}'.format(maxDepth),end="")
    for metric in range(len(trainPredErrors[maxDepth-1])):
        print('{:<15}'.format(str(np.round(trainPredErrors[maxDepth-1][metric],4))),end="")
    print("\n")
print("Test Data set:")
for maxDepth in dataConfig.dataSets[dataSetName]["maxTreeDepths"]:
    print('{:<15}'.format(maxDepth),end="")
    for metric in range(len(testPredErrors[maxDepth-1])):
        print('{:<15}'.format(str(np.round(testPredErrors[maxDepth-1][metric],4))),end="")
    print("\n")



    