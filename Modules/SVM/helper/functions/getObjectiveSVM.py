from Modules.helper.imports.packageImports import np

def getObjectiveSVM(weights, X, Y, hyperparameters={}):
    C = hyperparameters.get("C")
    N = hyperparameters.get("N")
    if C == None:
        C = 1
    if N == None:
        N = len(X)

    loss = (1/2)*(np.dot(weights[:-1].T, weights[:-1])) + C*N*np.sum(np.maximum(np.zeros((len(Y),)),1-(2*Y-1)*np.dot(X,weights)))

    return loss