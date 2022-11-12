from Modules.Perceptron.helper.imports.packageImports import *
from Modules.Perceptron.helper.imports.functionImports import *
from Modules.Perceptron.helper.imports.configImports import *

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

parser.add_argument("-pa",
                    "--perceptronAlgorithm",
                    choices=["standard", "voted", "averaged", "gaussianKernel"],
                    default="standard",
                    help="Perceptron algorithm to use",
                )

parser.add_argument("-r",
                    "--learningRate",
                    type=float, 
                    nargs = "+",
                    help="Learning Rate for GD",
                )

parser.add_argument("-T",
                    "--numEpochs",
                    type=int, 
                    help="Number of epochs",
                )

args = parser.parse_args()
if args.dataset:
    dataSetName = args.dataset
else: 
    print(f"No dataset provided!")
    exit(0)

miscConfig.handleMissingVals = args.handleMissingVals
debug = args.debug

validationSplit = 0.3
miscConfig.shuffleData = args.shuffle
if args.perceptronAlgorithm:
    miscConfig.perceptronAlgorithm = args.perceptronAlgorithm
if args.numEpochs:
    miscConfig.numEpochs = args.numEpochs
if args.learningRate:
    miscConfig.learningRate = args.learningRate
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

if dataConfig.dataSets[dataSetName]["attrVals"] == None:
    dataConfig.dataSets[dataSetName]["attrVals"] = getAttrVals(X)

if miscConfig.handleMissingVals:
    X, dataConfig.dataSets[dataSetName]["attrVals"] = handleMissingAttrValues(X,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"])
if miscConfig.handleMissingVals:
    testX, _ = handleMissingAttrValues(testX,dataConfig.dataSets[dataSetName]["categoricalAttrs"],dataConfig.dataSets[dataSetName]["attrVals"],False)
    
#Temporary
weights = np.ones((len(X),))
weights /= np.sum(weights)
#Temporary
X = X.astype(np.float64)
weights = weights.astype(np.float64)
Y = Y.astype(np.float64)

testX = testX.astype(np.float64)
testY = testY.astype(np.float64)

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

bestErr = np.inf
bestLR = None

print(f"Perceptron ({miscConfig.perceptronAlgorithm}):")
for lr in miscConfig.learningRate:
    print(f"\tLearning Rate = {lr}")
    model = fit(train_X, train_W, train_Y, lr, miscConfig.numEpochs, miscConfig.perceptronAlgorithm, debug)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print("\t\t-----------------------")
    # print(f"\t\tLearned model:")
    # for key in model.keys():
    #         print(f"\t\t{key}:\n\t\t\t{model[key]}")
    preds = predict(train_X, model)
    err = getError(preds, train_Y)

    print(f"\t\tTrain Error = {err}")
    if not args.validate and err < bestErr:
        bestErr = err 
        bestLR = lr

    if args.validate:
        preds = predict(val_X, model)
        err = getError(preds, val_Y, val_W)
        print(f"\t\tValidation Error = {err}")
        if err < bestErr:
            bestErr = err 
            bestLR = lr

    preds = predict(testX, model)
    errTest = getError(preds, testY)
    print(f"\t\tTest Error = {errTest}")
    print("\t----------------------------------------------")

if len(miscConfig.learningRate) > 1:
    print("---------------------------------------------------------------------")
    print(f"Best learning rate = {bestLR}, Best loss = {bestErr}")