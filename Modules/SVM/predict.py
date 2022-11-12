from Modules.SVM.helper.imports.packageImports import *
from Modules.SVM.helper.imports.classImports import *
from Modules.SVM.helper.imports.functionImports import *
from Modules.SVM.helper.imports.configImports import *

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

parser.add_argument("-gamma",
                    type=float, 
                    nargs = "+",
                    help="Gamma for Gaussian Kernel for dual form of SVM",
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

parser.add_argument("-a",
                    type=float, 
                    nargs = "+",
                    help="Value of a in learning rate decomposition scheme 1",
                )

parser.add_argument("-C",
                    type=float, 
                    nargs = "+",
                    help="Value of C in SVM objective",
                )

parser.add_argument("-form",
                    choices=["primal", "dual"],
                    default="primal",
                    help="Form of SVM objective function to use",
                )

parser.add_argument("-kernel",
                    choices=["gaussian"],
                    help="Type of kernel to use for dual form of objective",
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
if args.C:
    miscConfig.C = args.C
if args.a:
    miscConfig.a = args.a
if args.form:
    miscConfig.form = args.form
if args.kernel:
    miscConfig.kernel = args.kernel
if args.gamma:
    miscConfig.gamma = args.gamma

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

miscConfig.N = len(train_X)

if not (miscConfig.form == "dual" and miscConfig.kernel == "gaussian"):
    miscConfig.gamma = [None] #Dummy gamma

for lr in miscConfig.learningRate:
    print(f"Learning Rate = {lr}")
    for C in miscConfig.C:
        print(f"\tC = {C}")
        for g in miscConfig.gamma:
            hyperparameters = {
                "C": C,
                "N": miscConfig.N,
                "a": miscConfig.a,
                "gamma": g
            }
            if miscConfig.form == "dual" and miscConfig.kernel == "gaussian":
                print(f"\t\tg = {g}")
                hyperparameters["gamma"] = g
            clf = SVM(form=miscConfig.form, kernel=miscConfig.kernel)
            clf.fit(train_X, train_Y, miscConfig.numEpochs, lr, miscConfig.lrScheme, hyperparameters, debug, savePlots)
            preds = clf.predict(train_X)
            err = getError(preds, train_Y)
            print(f"\t\t\tTraining error: {err}")
            preds = clf.predict(test_X)
            err = getError(preds, test_Y)
            print(f"\t\t\tTest error: {err}")