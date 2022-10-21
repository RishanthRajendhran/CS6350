from helper.imports.packageImports import np, preprocessing
from Modules.EnsembleLearning.AdaBoost.helper.functions.trainLearner import trainLearner
from Modules.EnsembleLearning.AdaBoost.helper.functions.makePredictions import makePredictions
from Modules.EnsembleLearning.AdaBoost.helper.functions.labelEncodeCol import labelEncodeCol
from Modules.EnsembleLearning.AdaBoost.helper.functions.labelDecodeCol import labelDecodeCol
from Modules.DecisionTree.helper.imports.functionImports import getError, getAccuracy

#performAdaBoost
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   T               :   Number of weak learners to train
#                   weakLearner     :   Name of weak learner
#                                       Choices: ["decisionTree"]
#                   conditions      :   Dictionary of conditions to satisfy while training 
#                                       weak learner
#                   debug           :   Boolean flag to indicate whether to print debug
#                                       print statements or not
#                                       Default: False
#Output         :
#                   Hs              :   List of trained weak learners
#                   alphas          :   List of weights of weak learners
#                   preds           :   List of predictions made by combined weak learners on X
#                   _               :   Prediction error made by combined weak learners on X
#                   _               :   Prediction accuracy of combined weak learners on X
#                   allErrs         :   List of errors made by all weak learners
#                   allAccs         :   List of accuracies made by all weak learners
#What it does   :
#                   This function is used to perform AdaBoost
def performAdaBoost(X, weights, Y, T, weakLearner, conditions, debug=False):
    if len(np.unique(Y)) != 2:
        print("Only binary classification supported!")
        return None, None, None, None, None, None, None
    #Weights of data instances 
    #Initialisation: Uniform
    D = weights
    alphas = []
    Hs = []
    allPreds = []
    allErrs = []
    allAccs = []
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(Y))
    for t in range(1,T+1):
        h = trainLearner(X, D, Y, weakLearner, conditions)
        Hs.append(h)
        preds, err, _ = makePredictions(X, D, Y, h, weakLearner)

        epsilon = err
        allErrs.append(epsilon)
        allAccs.append(1-epsilon)


        if epsilon == 0:
            epsilon = 0.0001
        elif epsilon == 1:
            epsilon -= 0.0001

        allPreds.append(preds)
        alpha = (1/2)*np.log((1-epsilon)/epsilon)
        alphas.append(alpha)

        # h.printTree()
        if debug:
            print(f"\tError made by weakLearner{t} on trainX: {epsilon}")
            print(f"\tAccuracy of weakLearner{t} on trainX: {1-epsilon}\n")

        alphas = np.array(alphas)
        Y = np.array(Y)
        preds = np.array(preds)

        D = D*np.exp(alpha*(Y!=preds))
        D /= np.sum(D)

        alphas = alphas.tolist()
        Y = Y.tolist()
        preds = preds.tolist()

        if np.round(epsilon,15) == 0: 
            break

    alphas = np.array(alphas)
    Hs = np.array(Hs)
    allPreds = np.array(allPreds)
    finalPreds = np.sign(np.sum((alphas*(2*labelEncodeCol(allPreds, le)-1).T).T,axis=0).astype(np.float64))
    preds = labelDecodeCol(((finalPreds+1)/2).astype(int),le)
    # for i in range(len(allPreds)):
    #     print(f"{i} => {np.sum(allPreds[i]==preds)/len(preds)}")
    # for i in range(len(preds[:20])):
    #     print(f"Prediction {i} = {preds[i]}")
    #     print(f"Actual {i} = {Y[i]}")
    #     for j in range(len(allPreds)):
    #         print(f"\tPrediction {j} = {allPreds[j][i]}")
    #     print("---------------------------")
    # print(np.sum(Y==preds))
    return Hs, alphas, preds, getError(Y, preds, weights), getAccuracy(Y, preds, weights), allErrs, allAccs