from Modules.LinearRegression.helper.imports.packageImports import *
from Modules.LinearRegression.helper.imports.functionImports import *
from Modules.LinearRegression.helper.imports.configImports import *

parser = argparse.ArgumentParser()
parser.add_argument("-ds",
                    "--dataset",
                    help="Data set to use for building learner",
                    choices=["concrete"]
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

parser.add_argument("-gd",
                    "--gradientDescent",
                    choices=["batch", "stochastic"],
                    default="batch",
                    help="Type of gradient descent to perform",
                )

parser.add_argument("-l",
                    "--loss",
                    choices=["lms"],
                    default="lms",
                    help="Type of loss to use",
                )

parser.add_argument("-lr",
                    "--learningRate",
                    type=float, 
                    nargs = "+",
                    help="Learning Rate for GD",
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
debug = args.debug

validationSplit = 0.3
saveResults = args.saveResults
miscConfig.shuffle = args.shuffle
miscConfig.gradientDescent = args.gradientDescent
miscConfig.loss = args.loss
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
miscConfig.conditions["attrVals"] = dataConfig.dataSets[dataSetName]["attrVals"]

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

train_X = X.copy()
train_W = weights.copy()
train_Y = Y.copy()

if args.validate:
    train_X, train_W, train_Y, val_X, val_W, val_Y = trainValSplit(X, weights, Y, miscConfig.validationSplit, miscConfig.shuffleData)

bestLoss = np.inf
bestLR = None

# from sklearn import linear_model
# reg = linear_model.LinearRegression()
# reg.fit(X,Y)
# preds = reg.predict(testX.astype(np.float64))
# print(preds[:20])
# print(Y[:20])
# print(np.sum((preds-testY.astype(np.float64))**2)/len(testY))
# exit(0)

print("Linear Regression:")
for lr in miscConfig.learningRate:
    print(f"\tLearning Rate = {lr}")
    w, allLosses = findLinearRegressor(train_X, train_W, train_Y, lr, miscConfig.gradientDescent, miscConfig.loss,debug)
    preds = makePredictions(train_X, w, True)
    err = getLoss(preds, train_W, train_Y, w, miscConfig.loss)

    visualizeResults(allLosses, saveResults, f"TrainLossLinearRegression{miscConfig.gradientDescent}{lr}")

    print(f"\t\tTrain Loss = {err}")
    if not args.validate and err < bestLoss:
        bestLoss = err 
        bestLR = lr

    if args.validate:
        preds = makePredictions(val_X, w, True)
        err = getLoss(preds, val_W, val_Y, w, miscConfig.loss)
        print(f"\t\tValidation Loss = {err}")
        if err < bestLoss:
            bestLoss = err 
            bestLR = lr

    preds = makePredictions(testX, w, True)
    errTest = getLoss(preds, np.ones((len(testX),1))/len(testX), testY, w, miscConfig.loss)
    print(f"\t\tTest Loss = {errTest}")

    #How close is the obtained weight vector to the ideal weight vector obtained analytically 
    newX = np.ones((train_X.shape[0],train_X.shape[1]+1))
    newX[:,1:] = train_X.copy()
    idealW = np.dot(np.linalg.pinv(np.dot(newX.T, newX)), np.dot(newX.T, Y))
    idealPreds = makePredictions(newX, idealW, False)
    idealLoss = getLoss(idealPreds, np.ones((len(newX),1))/len(newX), train_Y, idealW, miscConfig.loss)
    idealPredsTest = makePredictions(testX, idealW, True)
    idealLossTest = getLoss(idealPredsTest, np.ones((len(testX),1))/len(testX), testY, idealW, miscConfig.loss)
    diffW = idealW - w
    diffW =  np.abs(diffW)
    print("\n\t\tAdditional information:")
    print(f"\t\t\tidealLoss = {idealLoss}")
    print(f"\t\t\tactualLoss = {err}")
    print(f"\t\t\tidealLossTest = {idealLossTest}")
    print(f"\t\t\tactualLossTest = {errTest}")
    print(f"\t\t\tidealW = {idealW}")
    print(f"\t\t\tw = {w}")
    print(f"\t\t\tdiffW = {diffW}")
    print(f"\t\t\t|diffW| = {np.linalg.norm(diffW)}\n")

print(f"Best learning rate = {bestLR}, Best loss = {bestLoss}")