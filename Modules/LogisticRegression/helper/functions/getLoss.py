from Modules.helper.imports.packageImports import np

def getLoss(weights, X, Y, hyperparameters={}):
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
        loss = 0
        if obj == "map":
            loss = np.dot(weights[:-1], weights[:-1])/variance
        loss += C*M*np.log(1+np.exp(-Y*np.dot(weights, X)))
    else:
        # print(np.dot(weights[:-1], weights[:-1])/variance)
        loss = 0
        if obj == "map":
            loss = np.dot(weights[:-1], weights[:-1])/variance
        loss += C*np.sum(np.log(1+np.exp(-Y*np.dot(X, weights))))

    return loss