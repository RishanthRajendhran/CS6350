from Modules.EnsembleLearning.Bagging.helper.imports.packageImports import *
from Modules.EnsembleLearning.Bagging.helper.imports.functionImports import *
from Modules.EnsembleLearning.Bagging.helper.imports.configImports import *

parser = argparse.ArgumentParser()
parser.add_argument("-ds",
                    "--dataset",
                    help="Data set to use for building learner",
                    choices=["bank", "creditCard", "iris"]
                )
parser.add_argument("-hmv",
                    "--handleMissingVals",
                    action="store_true",
                    help="Boolean flag indicating whether missing values need to be imputed"
                )

parser.add_argument("-l",
                    "--learner",
                    help="Name of learner to learn",
                    choices=miscConfig.allowedLearners,
                    default="decisionTree"
                )

parser.add_argument("-nl",
                    "--numLearners",
                    type=int,
                    help="Number of learners to learn"
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
                    "--tuneNumLearners",
                    action="store_true",
                    help="Boolean flag to indicate if numLearners tuning needs to be done",
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

parser.add_argument("-fbs",
                    "--fracBootstrapSamples",
                    type=float,
                    help="Fraction (>0 and <=1) of data instances to use as bootstrap samples for bagging",
                )

parser.add_argument("-m",
                    "--numBootstrapSamples",
                    type=float,
                    help="Number of data instances to use as bootstrap samples for bagging",
                )

parser.add_argument("-rf",
                    "--randomForest",
                    action="store_true",
                    help="Boolean flag to indicate whether to grow a random forest",
                )

parser.add_argument("-fs",
                    "--fracAttrs",
                    type=float,
                    help="Fraction (>0 and <=1) of attributes to choose for Random Forest",
                )

parser.add_argument("-g",
                    "--numAttrs",
                    type=int,
                    help="Number of attributes to choose for Random Forest",
                )

parser.add_argument("-nal",
                    "--numAttrsList",
                    type=int,
                    nargs = "+",
                    help="List of number of attributes to pick for Random Forest"
                )

parser.add_argument("-exp",
                    "--experiment",
                    action="store_true",
                    help="Boolean flag to indicate whether to switch to experiment mode",
                )
        
args = parser.parse_args()
if args.dataset:
    dataSetName = args.dataset
else: 
    print(f"No dataset provided!")
    exit(0)

miscConfig.handleMissingVals = args.handleMissingVals

miscConfig.learner = args.learner

if args.numLearners:
    miscConfig.numLearners = args.numLearners

debug = args.debug
tune = args.tuneNumLearners
saveResults = args.saveResults

validationSplit = 0.3
miscConfig.shuffleData = args.shuffle
if args.validationSplit:
    miscConfig.validationSplit = args.validationSplit

train = readCSV(dataConfig.dataSets[dataSetName]["trainPath"], dataConfig.dataSets[dataSetName]["skipHeader"])
test = readCSV(dataConfig.dataSets[dataSetName]["testPath"], dataConfig.dataSets[dataSetName]["skipHeader"])

if args.numValInstances and len(train) >= args.numValInstances:
    miscConfig.validationSplit = args.numValInstances/len(train)

X = train[:,:-1].copy()
Y = train[:,-1].copy()
testX = test[:,:-1].copy()
testY = test[:,-1].copy()

if args.fracBootstrapSamples:
    miscConfig.fracBootstrapSamples = args.fracBootstrapSamples
    miscConfig.m = int(len(X)*miscConfig.fracBootstrapSamples)

if args.numBootstrapSamples:
    miscConfig.m = args.numBootstrapSamples

miscConfig.randomForest = args.randomForest
if args.fracAttrs:
    miscConfig.fracAttrs = args.fracAttrs
    miscConfig.g = int(X.shape[1]*miscConfig.fracAttrs)

if args.numAttrs:
    miscConfig.g = args.numAttrs

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

X, dataConfig.dataSets[dataSetName]["attrVals"] = convertNumericalToCategorical(X,dataConfig.dataSets[dataSetName]["numericalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
if miscConfig.handleMissingVals:
    X, dataConfig.dataSets[dataSetName]["attrVals"] = handleMissingAttrValues(X,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
testX, _ = convertNumericalToCategorical(testX,dataConfig.dataSets[dataSetName]["numericalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"],False)
if miscConfig.handleMissingVals:
    testX, _ = handleMissingAttrValues(testX,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"],False)

#Temporary
weights = np.ones((len(X),))
weights /= np.sum(weights)
#Temporary

if args.validate:
    train_X, train_W, train_Y, val_X, val_W, val_Y = trainValSplit(X, weights, Y, miscConfig.validationSplit, miscConfig.shuffleData)
    X = train_X.copy()
    weights = train_W.copy()
    Y = train_Y.copy()

NLs = [miscConfig.numLearners]

if tune:
    NLs = np.arange(miscConfig.numLearners)+1

if miscConfig.g == None and not miscConfig.randomForest:
    miscConfig.g = X.shape[1]

if miscConfig.m == None:
    miscConfig.m = int(len(X)*miscConfig.fracBootstrapSamples)

if args.numAttrsList:
    miscConfig.numAttrsList = args.numAttrsList
elif args.numAttrs:
    miscConfig.numAttrsList = [args.numAttrs]
else:
    miscConfig.numAttrsList = [miscConfig.g]

if miscConfig.randomForest:
    print("Growing a random forest...")
else:
    print("Bagging...")

if args.experiment:
    if miscConfig.g == None:
        miscConfig.g = X.shape[1]
    print("Experimenting...")
    # allResults = {}
    allHs = []
    for i in range(100):
        print(f"Iteration {i} of 100...")
        miscConfig.conditions["replacement"] = False
        Hs, preds, err, acc, allErrs, allAccs = performBagging(X, weights, Y, 1000, 500, miscConfig.learner, miscConfig.conditions, miscConfig.randomForest, miscConfig.g, debug)
        # allResults[i] = {
        #     "Hs": Hs,
        #     "preds":preds,
        #     "err":err,
        #     "acc":acc,
        #     "allErrs":allErrs,
        #     "allAccs":allAccs
        # }
        allHs.append(Hs)
    bias = 0
    variance = 0
    for i in range(len(testX)):
        allPreds = []
        for j in range(100):
            h = allHs[j][0]
            pred = h.predict(testX[i])
            if pred == "no":
                allPreds.append(-1)
            else:
                allPreds.append(1)
        avgPred = np.sign(np.sum(allPreds))
        if testY[i] == "no":
            actualLabel = -1
        else:
            actualLabel = 1
        
        bias += (actualLabel - avgPred)**2

        var = 0
        for pred in allPreds:
            var += (avgPred-pred)**2
        var /= (100-1)
        variance += var
    bias /= len(testX)
    variance /= len(testX)
    gse = bias + variance 
    print(f"Single tree learner over testX:")
    print(f"\tBias = {bias}")
    print(f"\tVariance = {variance}")
    print(f"\tGeneral Squared Error = {gse}")
    #-----------------------------------------
    allPreds = []
    for i in range(100):
        preds, _, _, _, _ = makeBaggedPredictions(testX, np.ones((len(testX),))/len(X), testY, allHs[i], miscConfig.learner)
        allPreds.append(preds)
    allPreds = np.array(allPreds)

    bias = 0
    variance = 0
    for i in range(len(testX)):
        v, c = np.unique(allPreds[:,i], return_counts=True)
        pred = v[np.argmax(c)]
        if pred == "no":
            avgPred = -1
        else:
            avgPred = 1

        if testY[i] == "no":
            actualLabel = -1
        else:
            actualLabel = 1
        
        bias += (actualLabel - avgPred)**2

        var = 0
        for j in range(100):
            if allPreds[j,i] == "no":
                var += (avgPred + 1)**2
            else:
                var += (avgPred - 1)**2
        var /= (100-1)
        variance += var
    
    bias /= len(testX)
    variance /= len(testX)
    gse = bias + variance 
    if miscConfig.randomForest:
        print(f"Random forest over testX:")
    else:
        print(f"Bagged learners over testX:")
    print(f"\tBias = {bias}")
    print(f"\tVariance = {variance}")
    print(f"\tGeneral Squared Error = {gse}")
    exit(0)

bestG = None
bestGNL = None
bestGErr = 1

bestGTest = None
bestGNLTes = None
bestGErrTest = 1

algo = "Bagging"
if miscConfig.randomForest:
    algo = "randomForest"
for g in miscConfig.numAttrsList:
    print(f"numAttrs = {g}")

    bestErr, bestNL = 1, None
    bestErrTest, bestNLTest = 1, None
    allHsErrs, allHsAccs = [], []
    allHsErrsTest = []

    for nl in NLs:
        print(f"\tnumLearners = {nl}")
        Hs, preds, err, acc, allErrs, allAccs = performBagging(X, weights, Y, miscConfig.m, nl, miscConfig.learner, miscConfig.conditions, miscConfig.randomForest, g, debug)
        allHsErrs.append(err)
        allHsAccs.append(acc)

        allTrainErrs = allErrs.copy()

        print(f"\t\tError made by {algo} on trainX: {err}")
        print(f"\t\tAccuracy of {algo} on trainX: {acc}")

        if not args.validate and bestErr >= err:
            bestErr = err 
            bestNL = nl

        if args.validate:
            print("\n\t\tPerforming validation...")
            _, err, acc, allErrs, allAccs = makeBaggedPredictions(val_X, val_W, val_Y, Hs, miscConfig.learner)
            print(f"\t\tError made by {algo} on validX: {err}")
            print(f"\t\tAccuracy of {algo} on validX: {acc}")
            
            visualizeResults(allErrs, saveResults, f"LearnerError{algo}Val{nl}_{g}", False)

            if bestErr >= err:
                bestErr = err 
                bestNL = nl
        
        print("\n\t\tPerforming testing...")
        _, err, acc, allErrs, allAccs = makeBaggedPredictions(testX, np.ones((len(testX),))/len(X), testY, Hs, miscConfig.learner)
        
        allHsErrsTest.append(err)
        
        print(f"\t\tError made by {algo} on testX: {err}")
        print(f"\t\tAccuracy of {algo} on testX: {acc}")
        visualizeResults(allErrs, saveResults, f"LearnerError{algo}{nl}_{g}", False)
        visualizeResults(allTrainErrs, saveResults, f"LearnerError{algo}{nl}_{g}", True)
        print("---------------------------")
        if bestErrTest >= err:
            bestErrTest = err 
            bestNLTest = nl
    
    if len(NLs) > 1:
        visualizeResults(allHsErrs, saveResults, f"{algo}Error{g}", True)
        visualizeResults(allHsErrsTest, saveResults, f"{algo}TestError{g}", True)
        print(f"(train data) => Best numLearners = {bestNL}\nError = {bestErr}\nBestAccuracy = {1-bestErr}")
        print(f"(test data) => Best numLearners = {bestNLTest}\nError = {bestErrTest}\nBestAccuracy = {1-bestErrTest}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
   
    if bestErr <= bestGErr:
        bestGErr = bestErr
        bestG = g
        bestGNL = bestNL 

    if bestErrTest <= bestGErrTest:
        bestGErrTest = bestErrTest
        bestGTest = g
        bestGNLTest = bestNLTest 

if len(miscConfig.numAttrsList) > 1:
    print(f"(train data) => Best numLearners = {bestGNL}\nBest numAttrs = {bestG}\nError = {bestGErr}\nBestAccuracy = {1-bestGErr}")
    print(f"(test data)  => Best numLearners = {bestGNLTest}\nBest numAttrs = {bestGTest}\nError = {bestGErrTest}\nBestAccuracy = {1-bestGErrTest}")