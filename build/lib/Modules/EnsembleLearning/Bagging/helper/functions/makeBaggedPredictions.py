from Modules.helper.imports.packageImports import np
from Modules.DecisionTree.helper.imports.functionImports import testTree, getError, getAccuracy
from Modules.EnsembleLearning.AdaBoost.helper.functions.makePredictions import makePredictions


#makeBaggedPredictions
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   Hs              :   List of trained weak learners h's
#                   learner         :   Name of learner h
#                                       Choices: ["decisionTree"]
#Output         :
#                   preds           :   Predictions made by AdaBoosted weak learners on X
#                   _               :   Prediction error
#                   _               :   Prediction accuracy
#                   allErrs         :   List of errors made by all weak learners
#                   allAccs         :   List of accuracies made by all weak learners
#What it does   :
#                   This function is used to make predictions using Bagged learners
def makeBaggedPredictions(X, weights, Y, Hs, learner):
    if learner == "decisionTree":
        allPreds = []
        allErrs = []
        allAccs = []
        for h in Hs:
            preds, err, acc = makePredictions(X, weights, Y, h, learner)
            allErrs.append(err)
            allAccs.append(acc)
            allPreds.append(preds)
        allPreds = np.array(allPreds)
        finalPreds = []
        for i in range(allPreds.shape[1]):
            v, c = np.unique(allPreds[:,i], return_counts=True)
            finalPreds.append(v[np.argmax(c)])
        predsAcc = getAccuracy(Y, preds, weights)
        predsErr = 1 - predsAcc
        return preds, predsErr, predsAcc, allErrs, allAccs
    else: 
        return None, None, None, None, None