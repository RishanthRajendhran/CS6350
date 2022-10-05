from helper.imports.packageImports import np, preprocessing
from Modules.DecisionTree.helper.imports.functionImports import testTree, getError, getAccuracy
from Modules.EnsembleLearning.AdaBoost.helper.functions.makePredictions import makePredictions
from Modules.EnsembleLearning.AdaBoost.helper.functions.labelEncodeCol import labelEncodeCol
from Modules.EnsembleLearning.AdaBoost.helper.functions.labelDecodeCol import labelDecodeCol

#makeAdaBoostPredictions
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   Hs              :   List of trained weak learners h's
#                   weakLearner     :   Name of weak learner h
#                                       Choices: ["decisionTree"]
#Output         :
#                   preds           :   Predictions made by AdaBoosted weak learners on X
#                   _               :   Prediction error
#                   _               :   Prediction accuracy
#What it does   :
#                   This function is used to make predictions using AdaBoosted weak learners
def makeAdaBoostPredictions(X, weights, Y, Hs, weakLearner):
    if weakLearner == "decisionTree":
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(Y))
        allPreds = []
        alphas = []
        for h in Hs:
            preds, err, acc = makePredictions(X, weights, Y, h, weakLearner)
            epsilon = np.sum(weights[np.where(Y!=preds)])
            if epsilon == 0:
                epsilon = 0.0001
            elif epsilon == 1:
                epsilon -= 0.0001

            alpha = (1/2)*np.log((1-epsilon)/epsilon)
            alphas.append(alpha)
            allPreds.append(preds)
        finalPreds = np.sign(np.sum((alphas*(2*labelEncodeCol(allPreds, le).T)-1).T,axis=0).astype(int))
        preds = labelDecodeCol(((finalPreds+1)/2).astype(int),le)
        epsilon = np.sum(weights[np.where(Y!=preds)])
        return preds, getError(Y, preds, weights), getAccuracy(Y, preds, weights)
    else: 
        return None, None, None