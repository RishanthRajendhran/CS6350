from Modules.Perceptron.helper.imports.packageImports import np
from Modules.DecisionTree.helper.functions.evaluation.getError import getError
from Modules.Perceptron.helper.functions.predict import predict

#fit
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   learningRate    :   Learning rate to use 
#                   T               :   Number of epochs 
#                   pa              :   Perceptron algorithm to use
#                                       Default: standard
#                   debug           :   Boolean flag to switch to debug mode
#                                       Default: False
#Output         :
#                   _               :   Dictionary containing description of perceptron model
#What it does   :
#                   This function is used to build a perceptron
def fit(X, weights, Y, learningRate, T, pa="standard", debug=False):
    if pa == "standard":
        w = np.zeros((X.shape[1],))
        b = 0
        for t in range(T):
            newOrder = np.random.permutation(len(X))
            for i in newOrder:
                pred = (np.sign(np.dot(X[i],w)+b)+1)/2
                if pred != Y[i]:
                    w = w + learningRate*(2*Y[i]-1)*X[i]
                    b = b + learningRate*(2*Y[i]-1)
            if debug and t%100==0:
                preds = predict(X, {"pa":"standard", "w":np.array(w), "b":np.array(b)})
                err = getError(Y, preds)
                print(f"\t\t\tIteration {t}")
                print(f"\t\t\tTrain Error = {err}")
                print("\t\t\t-----------------------")
        return {"pa":"standard", "w":np.array(w),"b":np.array(b)}
    elif pa == "voted":
        w = np.zeros((X.shape[1],))
        b = 0
        Cs = []
        Ws = []
        Bs = []
        c = 1
        for t in range(T):
            for i in range(len(X)):
                pred = (np.sign(np.dot(X[i],w)+b)+1)/2
                if pred != Y[i]:
                    Ws.append(w)
                    Bs.append(b)
                    Cs.append(c)
                    w = w + learningRate*(2*Y[i]-1)*X[i]
                    b = b + learningRate*(2*Y[i]-1)
                    c = 1
                else:
                    c = c + 1
            if debug and t%100==0:
                preds = predict(X, {"pa":"voted", "Ws":np.array(Ws), "Bs": np.array(Bs), "Cs":np.array(Cs)})
                err = getError(Y, preds)
                print(f"\t\t\tIteration {t}")
                print(f"\t\t\tTrain Error = {err}")
                print("\t\t\t-----------------------")
        return {"pa":"voted","Ws":np.array(Ws), "Bs": np.array(Bs), "Cs":np.array(Cs)}
    elif pa == "averaged":
        w = np.zeros((X.shape[1],))
        b = 0
        a = np.zeros((X.shape[1],))
        z = 0
        for t in range(T):
            for i in range(len(X)):
                pred = (np.sign(np.dot(X[i],w)+b)+1)/2
                if pred != Y[i]:
                    w = w + learningRate*(2*Y[i]-1)*X[i]
                    b = b + learningRate*(2*Y[i]-1)
                a = a + w
                z = z + b
            if debug and t%100==0:
                preds = predict(X, {"pa": "averaged", "a": np.array(a), "z": np.array(z)})
                err = getError(Y, preds)
                print(f"\t\t\tIteration {t}")
                print(f"\t\t\tTrain Error = {err}")
                print("\t\t\t-----------------------")
        return {"pa":"averaged", "a":np.array(a), "z": np.array(z)}
    return None
