from Modules.NeuralNetworks.helper.imports.packageImports import *
from Modules.NeuralNetworks.helper.imports.classImports import *
from Modules.NeuralNetworks.helper.imports.functionImports import *
from Modules.NeuralNetworks.helper.imports.configImports import *

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

parser.add_argument("-width",
                    type=int, 
                    nargs = "+",
                    help="Number of neurons in the hidden layer",
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

parser.add_argument("-summary",
                    action="store_true",
                    help="Boolean flag to indicate whether to print model summary",
                )

parser.add_argument("-wInit",
                    "--weightsInitialization",
                    choices=["zeros", "gaussian"],
                    default="gaussian",
                    help="Scheme for initialization of weights",
                )

parser.add_argument("-tensorflow",
                    action="store_true",
                    help="Boolean flag to indicate whether to use tensorflow to build neural network",
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
summary = args.summary
weightsInitialization = args.weightsInitialization
tensorflow = args.tensorflow

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
if args.width:
    miscConfig.width = args.width

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
if tensorflow:
    for depth in [3, 5, 9]:
        print(f"depth: {depth}")
        for width in [5, 10, 25, 50, 100]:
            print(f"\twidth: {width}")
            for activation in ["relu", "tanh"]:
                print(f"\t\tactivation: {activation}")
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                if activation == "relu":
                    weightInitializer = tf.keras.initializers.GlorotNormal()
                else: 
                    weightInitializer = tf.keras.initializers.HeUniform()
                clf = tf.keras.Sequential()
                for d in range(depth):
                    clf.add(
                        tf.keras.layers.Dense(width,
                            activation=activation, 
                            kernel_initializer=weightInitializer
                        ),
                    )
                clf.add(
                    tf.keras.layers.Dense(1, 
                        activation="linear", 
                        kernel_initializer=weightInitializer
                    ),
                )

                clf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics="accuracy"
                )

                verbosity = 0 
                if debug:
                    verbosity = 1

                clf.fit(train_X, train_Y, epochs=50, verbose=verbosity)

                trainLoss, trainAcc = clf.evaluate(train_X, train_Y, verbose=verbosity)
                print(f"\t\t\tTrain set: loss: {round(trainLoss,4)} error: {round(1-trainAcc,4)}")

                testLoss, testAcc = clf.evaluate(test_X, test_Y, verbose=verbosity)
                print(f"\t\t\tTest set: loss: {round(testLoss,4)} error: {round(1-testAcc,4)}")
                if bestConfig == None or bestConfig["trainError"] > (1-trainAcc):
                    bestConfig = {
                        "width": width,
                        "depth": depth,
                        "activation": activation,
                        "trainLoss": trainLoss,
                        "testLoss": testLoss,
                        "trainError": 1-trainAcc,
                        "testError": 1-testAcc,
                    }
    print(bestConfig)
else:   
    # NN = NeuralNetwork([
    #     Dense(2,
    #         activation="sigmoid",
    #         weights=[[-1, -2, -3], [1, 2, 3]]
    #     ),
    #     Dense(2,
    #         activation="sigmoid",
    #         weights=[[-1, -2, -3], [1, 2, 3]]
    #     ),
    #     Dense(1,
    #         activation="linear",
    #         weights=[[-1, 2, -1.5]]
    #     ),
    # ])

    # y, allActivations = NN([[1, 1]], return_activations=True)

    # grads = NN.backPropagate([1], allActivations)

    # print(grads)

    for w in miscConfig.width:
        print(f"Width: {w}")
        for lr in miscConfig.learningRate:
            hyperparameters = {
                "a": miscConfig.a,
                "width": w,
                "wInit": weightsInitialization,
            }
            print(f"\tLearning rate: {lr}")
            NN = NeuralNetwork()
            NN.add(Dense(w, 
                    activation="sigmoid", 
                    inputShape=train_X[0].shape,
                    weightsInitialization=weightsInitialization,
                ))
            NN.add(Dense(w, 
                    activation="sigmoid",
                    weightsInitialization=weightsInitialization,
                ))
            NN.add(Dense(1,
                weightsInitialization=weightsInitialization,
                ))

            if summary:
                NN.summary()

            np.set_printoptions(suppress=True)

            NN.train(train_X, train_Y, T=miscConfig.numEpochs, lr=lr, lrScheme=miscConfig.lrScheme, hyperparameters=hyperparameters, debug=debug, savePlot=savePlots)
            preds = NN(train_X)
            preds = np.ndarray.flatten(preds)
            trainLoss = np.sum((preds-train_Y)**2/2)
            trainErr = getError(np.rint(preds), train_Y)
            print(f"\t\t\tTrain set: loss: {trainLoss}, error: {trainErr}")
            preds = NN(test_X)
            preds = np.ndarray.flatten(preds)
            testLoss = np.sum((preds-test_Y)**2/2)
            testErr = getError(np.rint(preds), test_Y)
            print(f"\t\t\tTest set: loss: {testLoss}, error: {testErr}")
            print("+++++++++++++++++++++++++")
            if bestConfig == None or bestConfig["trainError"] > trainErr:
                bestConfig = {
                    "width": w,
                    "learningRate": lr,
                    "trainLoss": trainLoss,
                    "testLoss": testLoss,
                    "trainError": trainErr,
                    "testError": testErr
                }
    print(bestConfig)