from Modules.DecisionTree.helper.imports.packageImports import np
from Modules.LinearRegression.helper.functions.getLoss import getLoss
from Modules.LinearRegression.helper.functions.getGradient import getGradient
from Modules.LinearRegression.helper.functions.makePredictions import makePredictions

#performGradientDescent
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   w               :   Numpy array of initial weights of linear regressor
#                   learningRate    :   Learning rate for GD    
#                   gd              :   Type of gradient descent to perform
#                                       Default: batch
#                   loss            :   Type of loss function to use
#                                       Default: lms
#                   debug           :   Boolean flag to switch to debug mode
#                                       Default: False
#Output         :
#                   w               :   Numpy array of weights of linear regressor after GD
#                   allLosses       :   Numpy array of loss at all iterations
#What it does   :
#                   This function is used to perform GD for linear regression
def performGradientDescent(X, weights, Y, w, learningRate, gd="batch", loss="lms", debug=False):
    if gd == "batch":
        print("\t\tPerforming Batch Gradient Descent")
        it = 0
        oldW = w
        allLosses = []
        while 1:
            grad = getGradient(X,weights,Y,w,loss)
            oldW = w.copy()
            w = w - learningRate*grad

            preds = makePredictions(X, w)
            l = getLoss(preds,weights,Y,w,loss)
            allLosses.append(l)
            if debug and it % 100000 == 0:
                print(f"\t\t\tIteration {it}")
                print(f"\t\t\t\tLoss = {l}")
            if np.sum(np.abs(oldW-w)) < 0.000001:
                break
            it += 1

        return w, np.array(allLosses)
    elif gd == "stochastic":
        print("\t\tPerforming Stochastic Gradient Descent")
        it = 0
        oldW = w
        allLosses = []
        lUnchanged = 0
        oldL = None
        while 1:
            sample = np.random.randint(0,len(X))
            grad = getGradient(X[sample:(sample+1),:],weights[sample:(sample+1)],Y[sample:(sample+1)],w,loss)
            oldW = w.copy()
            w = w - learningRate*grad

            preds = makePredictions(X, w)
            l = getLoss(preds,weights,Y,w,loss)
            allLosses.append(l)
            if debug and it % 100000 == 0:
                print(f"\t\t\tIteration {it}")
                print(f"\t\t\t\tLoss = {l}")
            if oldL and abs(oldL - l) < 0.000001:
                lUnchanged += 1
            if lUnchanged > 2000000 or it > 1000000:
                break
            it += 1
            oldL = l
        return w, np.array(allLosses)
    return None, None
