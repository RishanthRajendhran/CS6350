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
#                   alphas          :   List of alphas
#                   weakLearner     :   Name of weak learner h
#                                       Choices: ["decisionTree"]
#                   allLabels       :   List of all unique target labels
#Output         :
#                   preds           :   Predictions made by AdaBoosted weak learners on X
#                   _               :   Prediction error
#                   _               :   Prediction accuracy
#                   allErrs         :   List of errors made by all weak learners
#                   allAccs         :   List of accuracies made by all weak learners
#What it does   :
#                   This function is used to make predictions using AdaBoosted weak learners
def makeAdaBoostPredictions(X, weights, Y, Hs, alphas, weakLearner, allLabels):
    if weakLearner == "decisionTree":
        le = preprocessing.LabelEncoder()
        le.fit(allLabels)
        allPreds = []
        allErrs = []
        allAccs = []
        for h in Hs:
            preds, err, acc = makePredictions(X, weights, Y, h, weakLearner)
            allErrs.append(err)
            allAccs.append(acc)
            allPreds.append(preds)

        alphas = np.array(alphas)
        allPreds = np.array(allPreds)
        finalPreds = np.sign(np.sum((alphas*(2*labelEncodeCol(allPreds, le)-1).T).T,axis=0).astype(np.float64))
        preds = labelDecodeCol(((finalPreds+1)/2).astype(int),le)

        predsAcc = getAccuracy(Y, preds, weights)
        predsErr = 1 - predsAcc

        return preds, predsErr, predsAcc, allErrs, allAccs
    else: 
        return None, None, None, None, None