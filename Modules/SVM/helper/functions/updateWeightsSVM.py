from Modules.helper.imports.packageImports import np

def updateWeightsSVM(x, y, weights, learningRate=0.01, hyperparameters={}):
    C = hyperparameters.get("C")
    N = hyperparameters.get("N")
    if C == None:
        C = 1
    if N == None:
        N = 1
    if (2*y-1)*np.dot(x, weights) <= 1:
        newW = np.zeros((weights.shape))
        newW[:-1] = weights[:-1].copy()
        return weights - learningRate*(newW - C*N*(2*y-1)*x)
    else:
        weights[:-1] = (1-learningRate)*weights[:-1]
        return weights