from helper.imports.packageImports import np, preprocessing
from Modules.EnsembleLearning.AdaBoost.helper.functions.trainWeakLeaner import trainWeakLeaner
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
#Output         :
#                   Hs              :   List of trained weak learners
#                   alphas          :   List of weights of weak learners
#                   preds           :   List of predictions made by combined weak learners on X
#                   _               :   Prediction error made by combined weak learners on X
#                   _               :   Prediction accuracy of combined weak learners on X
#What it does   :
#                   This function is used to perform AdaBoost
def performAdaBoost(X, weights, Y, T, weakLearner, conditions, debug=False):
    if len(np.unique(Y)) != 2:
        print("Only binary classification supported!")
        return None, None, None, None, None
    #Weights of data instances 
    #Initialisation: Uniform
    D = weights
    alphas = []
    Hs = []
    allPreds = []
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(Y))
    for t in range(1,T+1):
        h = trainWeakLeaner(X, D, Y, weakLearner, conditions)
        Hs.append(h)
        preds, err, acc = makePredictions(X, D, Y, h, weakLearner)

        epsilon = np.sum(D[np.where(Y!=preds)])
        if epsilon == 0:
            epsilon = 0.0001
        elif epsilon == 1:
            epsilon -= 0.0001

        allPreds.append(preds)
        alpha = (1/2)*np.log((1-epsilon)/epsilon)
        alphas.append(alpha)

        if debug:
            print(f"\tError made by weakLearner{t} on trainX: {epsilon}")
            print(f"\tAccuracy of weakLearner{t} on trainX: {1-epsilon}\n")

        alphas = np.array(alphas)
        Y = np.array(Y)
        preds = np.array(preds)

        D = D*np.exp(-alpha*(2*(Y==preds)-1))
        D /= np.sum(D)

        alphas = alphas.tolist()
        Y = Y.tolist()
        preds = preds.tolist()

        if np.round(epsilon,10) == 0: 
            break

    alphas = np.array(alphas)
    Hs = np.array(Hs)
    allPreds = np.array(allPreds)
    finalPreds = np.sign(np.sum((alphas*(2*labelEncodeCol(allPreds, le).T)-1).T,axis=0).astype(int))
    preds = labelDecodeCol(((finalPreds+1)/2).astype(int),le)
    # for i in range(len(allPreds)):
    #     print(f"{i} => {np.sum(allPreds[i]==preds)/len(preds)}")
    return Hs, alphas, preds, getError(Y, preds, weights), getAccuracy(Y, preds, weights)