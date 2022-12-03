from Modules.helper.imports.packageImports import np

def getGradient(weights, X, Y, hyperparameters):
    variance = 1
    if "variance" in hyperparameters.keys():
        variance = hyperparameters["variance"]
    gradientType = "sgd"
    if "gradientType" in hyperparameters.keys():
        gradientType = hyperparameters["gradientType"]
    C = 1
    if "C" in hyperparameters.keys():
        C = hyperparameters["C"]
    obj = "map"
    if "obj" in hyperparameters.keys():
        obj = hyperparameters["obj"]
    if gradientType == "sgd":    
        M = 1
        if "M" in hyperparameters.keys():
            M = hyperparameters["M"]
        grad = 0
        if obj == "map":
            w = weights.copy()
            w[:-1] *= 2
            w[:-1] /= variance
            grad = w 
        grad += C*M*((1/(1+np.exp(-Y*np.dot(X, weights))))-1)*Y*X
    else: 
        grad = 0
        if obj == "map":
            w = weights.copy()
            w[:-1] *= 2
            w[:-1] /= variance
            grad = w 
        grad += C*np.sum(((1/(1+np.exp(-Y*np.dot(X, weights))))-1)*Y*X)
    
    # print(grad)

    return grad