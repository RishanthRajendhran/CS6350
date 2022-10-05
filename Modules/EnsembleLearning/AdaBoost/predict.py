from Modules.EnsembleLearning.AdaBoost.helper.imports.packageImports import *
from Modules.EnsembleLearning.AdaBoost.helper.imports.functionImports import *
from Modules.EnsembleLearning.AdaBoost.helper.imports.configImports import *

parser = argparse.ArgumentParser()
parser.add_argument("-ds",
                    "--dataset",
                    help="Data set to use for building decision tree",
                    choices=["bank"]
                )
parser.add_argument("-hmv",
                    "--handleMissingVals",
                    action="store_true",
                    help="Boolean flag indicating whether missing values need to be imputed"
                )

parser.add_argument("-wl",
                    "--weakLearner",
                    help="Name of weak learner to learn",
                    choices=miscConfig.allowedWeakLearners,
                    default="decisionTree"
                )

parser.add_argument("-nwl",
                    "--numWeakLearners",
                    type=int,
                    help="Number of weak learners to learn"
                )

parser.add_argument("-debug",
                    action="store_true",
                    help="Boolean flag to indicate whether running in debug mode"
                )

parser.add_argument("-v",
                    "--validate",
                    action="store_true",
                    help="Boolean flag to indicate if train-validation split needs to be performed",
                )
        
args = parser.parse_args()
if args.dataset:
    dataSetName = args.dataset
else: 
    print(f"No dataset provided!")
    exit(0)

miscConfig.handleMissingVals = args.handleMissingVals

miscConfig.weakLearner = args.weakLearner

if args.numWeakLearners:
    miscConfig.numWeakLearners = args.numWeakLearners

debug = args.debug

train = readCSV(dataConfig.dataSets[dataSetName]["trainPath"], True)
test = readCSV(dataConfig.dataSets[dataSetName]["testPath"], True)

X = train[:,:-1].copy()
Y = train[:,-1].copy()
testX = test[:,:-1].copy()
testY = test[:,-1].copy()

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
if dataConfig.dataSets[dataSetName]["attrVals"] == None:
    dataConfig.dataSets[dataSetName]["attrVals"] = getAttrVals(X)
    
#Temporary
weights = np.ones((len(X),))
weights /= np.sum(weights)
#Temporary

if args.validate:
    train_X, train_W, train_Y, val_X, val_W, val_Y = trainValSplit(X, weights, Y, miscConfig.validationSplit, miscConfig.shuffleData)

conditions = {
    "maxDepth": 2
}
Hs, alphas, preds, err, acc = performAdaBoost(X, weights, Y, miscConfig.numWeakLearners, miscConfig.weakLearner, conditions, debug)

if debug:
    print(f"Error made by AdaBoost on trainX: {err}")
    print(f"Accuracy of AdaBoost on trainX: {acc}")

if args.validate:
    print("\nPerforming validation...")
    _, err, acc = makeAdaBoostPredictions(val_X, val_W, val_Y, Hs, miscConfig.weakLearner)
    print(f"Error made by AdaBoost on validX: {err}")
    print(f"Accuracy of AdaBoost on validY: {acc}")