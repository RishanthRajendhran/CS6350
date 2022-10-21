from Modules.helper.imports.packageImports import np
from Modules.EnsembleLearning.Bagging.helper.functions.makeBootstrapSamples import makeBootstrapSamples 
from Modules.EnsembleLearning.AdaBoost.helper.functions.trainLearner import trainLearner
from Modules.EnsembleLearning.AdaBoost.helper.functions.makePredictions import makePredictions
from Modules.DecisionTree.helper.imports.functionImports import getError, getAccuracy

#performBagging
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   m               :   Number of bootstrap samples to use
#                   T               :   Number of learners to train
#                   learner         :   Name of learner
#                                       Choices: ["decisionTree"]
#                   conditions      :   Dictionary of conditions to satisfy while training 
#                                       learner
#                   randomForest    :   Boolean flag to switch to RandomForest 
#                                       mode
#                                       Default: False
#                   g               :   Number of attributes to select for 
#                                       RandomForest
#                                       Default: None
#                   debug           :   Boolean flag to indicate whether to print debug
#                                       print statements or not
#                                       Default: False
#Output         :
#                   Hs              :   List of trained weak learners
#                   preds           :   List of predictions made by bagged learners on X
#                   _               :   Prediction error made by bagged learners on X
#                   _               :   Prediction accuracy of bagged learners on X
#                   allErrs         :   List of errors made by all learners
#                   allAccs         :   List of accuracies made by all learners
#What it does   :
#                   This function is used to perform Bagging
def performBagging(X, weights, Y, m, T, learner, conditions, randomForest=False, g=None, debug=False):
    if len(np.unique(Y)) != 2:
        print("Only binary classification supported!")
        return None, None, None, None, None, None, None
    Hs = []
    allPreds = []
    allErrs = []
    allAccs = []
    for t in range(1,T+1):
        drawnRows, drawnCols = makeBootstrapSamples(X, weights, m, randomForest, g, conditions["replacement"])
        newX, newW, newY = X[drawnRows, :], weights[drawnRows], Y[drawnRows]
        conditions["attrsRem"] = drawnCols 
        h = trainLearner(newX, newW, newY, learner, conditions)
        Hs.append(h)
        preds, err, acc = makePredictions(X, weights, Y, h, learner)

        allErrs.append(err)
        allAccs.append(acc)

        allPreds.append(preds)

        if debug:
            print(f"\tError made by learner{t} on trainX: {err}")
            print(f"\tAccuracy of learner{t} on trainX: {acc}\n")

    Hs = np.array(Hs)
    allPreds = np.array(allPreds)
    finalPreds = []
    for i in range(allPreds.shape[1]):
        v, c = np.unique(allPreds[:,i], return_counts=True)
        finalPreds.append(v[np.argmax(c)])
    return Hs, preds, getError(Y, preds, weights), getAccuracy(Y, preds, weights), allErrs, allAccs