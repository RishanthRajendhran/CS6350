from Modules.EnsembleLearning.AdaBoost.helper.imports.packageImports import *
from Modules.EnsembleLearning.AdaBoost.helper.imports.functionImports import *
from Modules.EnsembleLearning.AdaBoost.helper.imports.configImports import *

parser = argparse.ArgumentParser()
parser.add_argument("-ds",
                    "--dataset",
                    help="Data set to use for building weak learner",
                    choices=["bank", "creditCard", "iris"]
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

parser.add_argument("-tune",
                    "--tuneNumWeakLearners",
                    action="store_true",
                    help="Boolean flag to indicate if numWeakLearners tuning needs to be done",
                )

parser.add_argument("-nvi",
                    "--numValInstances",
                    type=int, 
                    help="Number of validation instances",
                )

parser.add_argument("-vs",
                    "--validationSplit",
                    type=float, 
                    help="Validation Split as a fraction",
                )

parser.add_argument("-shuffle",
                    action="store_true",
                    help="Boolean flag to indicate whether to shuffle the train data",
                )

parser.add_argument("-sr",
                    "--saveResults",
                    action="store_true",
                    help="Boolean flag to indicate whether to save visualizations of results",
                )
        
args = parser.parse_args()
if args.dataset:
    dataSetName = args.dataset
else: 
    print(f"No dataset provided!")
    exit(0)

miscConfig.handleMissingVals = args.handleMissingVals

miscConfig.weakLearner = args.weakLearner

miscConfig.allLabels = dataConfig.dataSets[dataSetName]["labelNames"]

if args.numWeakLearners:
    miscConfig.numWeakLearners = args.numWeakLearners

debug = args.debug
tune = args.tuneNumWeakLearners
saveResults = args.saveResults

validationSplit = 0.3
miscConfig.shuffleData = args.shuffle
if args.validationSplit:
    miscConfig.validationSplit = args.validationSplit

train = readCSV(dataConfig.dataSets[dataSetName]["trainPath"], True)
test = readCSV(dataConfig.dataSets[dataSetName]["testPath"], True)

if args.numValInstances and len(train) >= args.numValInstances:
    miscConfig.validationSplit = args.numValInstances/len(train)

X = train[:,:-1].copy()
Y = train[:,-1].copy()
testX = test[:,:-1].copy()
testY = test[:,-1].copy()

if dataSetName == "iris":
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    X = np.array(X).astype(str)
    Y = np.array(["yes" if Y[i]==1 else "no" for i in range(len(Y))])
    testX = X.copy()
    testY = Y.copy()

if dataConfig.dataSets[dataSetName]["attrVals"] == None:
    dataConfig.dataSets[dataSetName]["attrVals"] = getAttrVals(X)
miscConfig.conditions["attrVals"] = dataConfig.dataSets[dataSetName]["attrVals"]

#Preprocessing train data 
# Converting numerical attributes into binary categorical attributes
# By setting  the median as the threshold 
X, dataConfig.dataSets[dataSetName]["attrVals"] = convertNumericalToCategorical(X,dataConfig.dataSets[dataSetName]["numericalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
#Replace missing values (marked with "?")
# with the mean/median/mode/fractional examples
if miscConfig.handleMissingVals:
    X, dataConfig.dataSets[dataSetName]["attrVals"] = handleMissingAttrValues(X,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])

#Preprocessing test data 
# Converting numerical attributes into binary categorical attributes
# By setting  the median as the threshold   
testX, _ = convertNumericalToCategorical(testX,dataConfig.dataSets[dataSetName]["numericalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"],False)
#Replace missing values (marked with "?")
# with the mean/median/mode/fractional examples
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

if args.validate:
    train_X, train_W, train_Y, val_X, val_W, val_Y = trainValSplit(X, weights, Y, miscConfig.validationSplit, miscConfig.shuffleData)
    X = train_X.copy()
    weights = train_W.copy()
    Y = train_Y.copy()

NWLs = [miscConfig.numWeakLearners]

if tune:
    NWLs = np.arange(miscConfig.numWeakLearners)+1

print("Adaboosting...")


bestErr, bestNWL = 1, None
allHsErrs, allHsAccs = [], []
allHsErrsTest, allHsAccsTest = [], []
for nwl in NWLs:
    print(f"numWeakLeaners = {nwl}")
    Hs, alphas, preds, err, acc, allErrs, allAccs = performAdaBoost(X, weights, Y, nwl, miscConfig.weakLearner, miscConfig.conditions, debug)
    allHsErrs.append(err)
    allHsAccs.append(acc)

    allTrainErrs = allErrs.copy()

    print(f"Error made by AdaBoost on trainX: {err}")
    print(f"Accuracy of AdaBoost on trainX: {acc}")


    if not args.validate and bestErr >= err:
        bestErr = err 
        bestNWL = nwl

    if args.validate:
        print("\nPerforming validation...")
        _, err, acc, allErrs, allAccs = makeAdaBoostPredictions(val_X, val_W, val_Y, Hs, alphas, miscConfig.weakLearner, miscConfig.allLabels)
        print(f"Error made by AdaBoost on validX: {err}")
        print(f"Accuracy of AdaBoost on validY: {acc}")
        
        visualizeResults(allErrs, saveResults, f"WeakLearnerErrorAdaBoostVal{nwl}", True)
        print("---------------------------")
        if bestErr > err:
            bestErr = err 
            bestNWL = nwl

    print("\nPerforming testing...")
    _, err, acc, allErrs, allAccs = makeAdaBoostPredictions(testX, np.ones((len(testX),))/len(testX), testY, Hs, alphas, miscConfig.weakLearner, miscConfig.allLabels)
    allHsErrsTest = allErrs.copy()
    allHsAccsTest = allAccs.copy()

    print(f"Error made by AdaBoost on testX: {err}")
    print(f"Accuracy of AdaBoost on testX: {acc}")
    visualizeResults(allHsErrsTest, saveResults, f"WeakLearnerErrorAdaBoost{nwl}", False)
    visualizeResults(allTrainErrs, saveResults, f"WeakLearnerErrorAdaBoost{nwl}", True)
    print("---------------------------")

if len(NWLs) > 1:
    visualizeResults(allHsErrsTest, saveResults, f"AdaboostError", False)
    visualizeResults(allHsErrs, saveResults, f"AdaboostError", True)
    print(f"Best numWeakLearners = {bestNWL}\nError = {bestErr}\nBestAccuracy = {1-bestErr}")