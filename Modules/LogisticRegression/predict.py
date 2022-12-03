from Modules.LogisticRegression.helper.imports.packageImports import *
from Modules.LogisticRegression.helper.imports.functionImports import *
from Modules.LogisticRegression.helper.imports.configImports import *

parser = argparse.ArgumentParser()
parser.add_argument("-ds",
                    "--dataset",
                    help="Data set to use for building learner",
                    choices=["bankNote"]
                )
parser.add_argument("-hmv",
                    "--handleMissingVals",
                    action="store_true",
                    help="Boolean flag indicating whether missing values need to be imputed"
                )

parser.add_argument("-debug",
                    action="store_true",
                    help="Boolean flag to indicate whether running in debug mode"
                )

parser.add_argument("-savePlots",
                    action="store_true",
                    help="Boolean flag to indicate whether to save plots when running in debug mode"
                )

parser.add_argument("-v",
                    "--validate",
                    action="store_true",
                    help="Boolean flag to indicate if train-validation split needs to be performed",
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

parser.add_argument("-r",
                    "--learningRate",
                    type=float, 
                    nargs = "+",
                    help="Learning Rate for SGD",
                )

parser.add_argument("-T",
                    "--numEpochs",
                    type=int, 
                    help="Number of epochs",
                )

parser.add_argument("-lrScheme",
                    choices=["constant", "scheme1", "scheme2"],
                    help="Scheme for learning rate decomposition",
                )

parser.add_argument("-d",
                    type=float, 
                    nargs = "+",
                    help="Value of d in learning rate decomposition scheme 1",
                )

parser.add_argument("-C",
                    type=float,
                    help="Value of C in learning objective",
                )

parser.add_argument("-variance",
                    type=float, 
                    nargs = "+",
                    help="Value of variance of Gaussian distribution from which weights are drawn",
                )

parser.add_argument("-obj",
                    type=str, 
                    choices = ["map","ml"],
                    default="map",
                    help="Objective to use",
                )

args = parser.parse_args()
if args.dataset:
    dataSetName = args.dataset
else: 
    print(f"No dataset provided!")
    exit(0)

miscConfig.handleMissingVals = args.handleMissingVals
debug = args.debug
savePlots = args.savePlots
miscConfig.obj = args.obj

validationSplit = 0.3
miscConfig.shuffleData = args.shuffle
if args.numEpochs:
    miscConfig.numEpochs = args.numEpochs
if args.learningRate:
    miscConfig.learningRate = args.learningRate
if args.validationSplit:
    miscConfig.validationSplit = args.validationSplit
if args.lrScheme:
    miscConfig.lrScheme = args.lrScheme
if args.d:
    miscConfig.a = args.d
if args.variance:
    miscConfig.variance = args.variance
if args.C:
    miscConfig.C = args.C

train = readCSV(dataConfig.dataSets[dataSetName]["trainPath"], dataConfig.dataSets[dataSetName]["skipHeader"])
test = readCSV(dataConfig.dataSets[dataSetName]["testPath"], dataConfig.dataSets[dataSetName]["skipHeader"])

if args.numValInstances and len(train) >= args.numValInstances:
    miscConfig.validationSplit = args.numValInstances/len(train)

X = train[:,:-1].copy()
Y = train[:,-1].copy()
test_X = test[:,:-1].copy()
test_Y = test[:,-1].copy()

if dataConfig.dataSets[dataSetName]["attrVals"] == None:
    dataConfig.dataSets[dataSetName]["attrVals"] = getAttrVals(X)

if miscConfig.handleMissingVals:
    X, dataConfig.dataSets[dataSetName]["attrVals"] = handleMissingAttrValues(X,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
if miscConfig.handleMissingVals:
    test_X, _ = handleMissingAttrValues(test_X,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"],False)
    
#Temporary
weights = np.ones((len(X),))
weights /= np.sum(weights)
#Temporary
X = X.astype(np.float64)
weights = weights.astype(np.float64)
Y = Y.astype(np.float64)

test_X = test_X.astype(np.float64)
test_Y = test_Y.astype(np.float64)

train_X = X.copy()
train_W = weights.copy()
train_Y = Y.copy()

if args.validate:
    train_X, train_W, train_Y, val_X, val_W, val_Y = trainValSplit(X, weights, Y, miscConfig.validationSplit, miscConfig.shuffleData)
    train_X = train_X.astype(np.float64)
    train_W = train_W.astype(np.float64)
    train_Y = train_Y.astype(np.float64)
    val_X = val_X.astype(np.float64)
    val_W = val_W.astype(np.float64)
    val_Y = val_Y.astype(np.float64)
bestConfig = None  

# train_X = np.array([
#     [0.5, -1, 0.3],
#     [-1, -2, -2],
#     [1.5, 0.2, -2.5]
# ])

# train_Y = np.array([
#     1,
#     -1,
#     1
# ])

# test_X = train_X
# test_Y = train_Y

print("Logistic Regression")
if miscConfig.obj == "map":
    print("Maximum Aposteriori Estimation")
else:
    print("Maximum Likelihood Estimation")
    miscConfig.variance = [1]

for v in miscConfig.variance:
    if miscConfig.obj == "map":
        print(f"Variance: {v}")
    for lr in miscConfig.learningRate:
        print(f"\tLearning rate: {lr}")
        for d in miscConfig.a:
            print(f"\t\td: {d}")
            hyperparameters = {
                "a": d,
                "gradType": "sgd",
                "M": len(train_X),
                "variance": v,
                "obj": miscConfig.obj,
                "C": miscConfig.C,
            }
            weights = performSGD(train_X, train_Y, updateWeights,  getLoss, miscConfig.numEpochs, lr, miscConfig.lrScheme,hyperparameters, debug, savePlots)
            preds = predict(weights, train_X)
            preds = np.ndarray.flatten(preds)
            preds = (preds+1)/2
            trainErr = getError(preds, train_Y)
            print(f"\t\t\tTrain set: error: {trainErr}")
            preds = predict(weights, test_X)
            preds = np.ndarray.flatten(preds)
            preds = (preds+1)/2
            testErr = getError(preds, test_Y)
            print(f"\t\t\tTest set: error: {testErr}")
            print("+++++++++++++++++++++++++")
            if bestConfig == None or bestConfig["trainError"] > trainErr:
                bestConfig = {
                    "variance": v,
                    "learningRate": lr,
                    "trainError": trainErr,
                    "testError": testErr
                }
print(bestConfig)