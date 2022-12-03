from Modules.LogisticRegression.helper.functions.getGradient import getGradient

def updateWeights(weights, X, Y, learningRate, hyperparameters):
    weights = weights - learningRate*getGradient(weights, X, Y, hyperparameters)

    return weights