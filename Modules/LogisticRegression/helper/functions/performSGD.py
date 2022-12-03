from Modules.helper.imports.packageImports import np, plt

import Modules.Results

def performSGD(X, Y, updateWeights, getLoss, T=100, lr=0.01, lrScheme="constant", hyperparameters={}, debug=False, savePlot = False, weights=[]):
    augX = np.ones((X.shape[0], X.shape[1]+1))
    augX[:,:-1] = X.copy()

    X = augX.copy()

    if len(weights) != X.shape[1]:
        weights = np.zeros((X.shape[1],))
    weights = np.array(weights)

    learningRate = lr

    allLosses = []
    for t in range(T):
        newOrder = np.random.permutation(len(X))
        newOrder = np.arange(len(X))
        for i in newOrder:
            x = X[i,:].copy()
            y = Y[i].copy()
            if lrScheme == "scheme1":
                learningRate = lr/(1+((lr/hyperparameters["a"])*t))
            elif lrScheme == "scheme2":
                learningRate = lr/(1+t)
            weights = updateWeights(weights, x, y, learningRate, hyperparameters)
            loss = getLoss(weights, x, y, hyperparameters)
        hyperparameters["gradientType"] = "overall"
        loss = getLoss(weights, X, Y, hyperparameters)
        hyperparameters["gradientType"] = "sgd"
        allLosses.append(loss)
        if debug and t%10==0:
            print(f"\t\t\t\tIteration {t+1}: Loss={loss}")
    allLosses = np.array(allLosses)
    
    if debug and T%10==0:
        hyperparameters["gradientType"] = "overall"
        loss = getLoss(weights, X, Y, hyperparameters)
        hyperparameters["gradientType"] = "sgd"
        print(f"\t\t\tEnd of training: Loss={loss}")

    # if debug or savePlot:
    #     plt.plot(allLosses)
    #     plt.xlabel("Iterations")
    #     plt.ylabel("Loss")
    #     if savePlot:
    #         hyps =  ""
    #         for key in hyperparameters.keys():
    #             hyps += "_"
    #             hyps += str(hyperparameters[key])
    #         plt.savefig(f"{str(list(Modules.Results.__path__)[0])}/LogisticLoss_{T}_{lr}_{lrScheme}{hyps}.png")
    #         plt.clf()
    #     else:
    #         plt.show()

    return weights


    