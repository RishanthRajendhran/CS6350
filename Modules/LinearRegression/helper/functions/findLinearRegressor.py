from Modules.DecisionTree.helper.imports.packageImports import np
from Modules.LinearRegression.helper.imports.functionImports import performGradientDescent

#findLinearRegressor
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy array of weights of data instances in X
#                   Y               :   Numpy array of target labels of instances in X
#                   learningRate    :   Learning rate to use for GD
#                   gd              :   Type of gradient descent to perform
#                                       Default: batch
#                   loss            :   Type of loss function to use
#                                       Default: "lms"
#                   debug           :   Boolean flag to switch to debug mode
#                                       Default: False
#Output         :
#                   w               :   Numpy array of weights of linear regressor
#                   allLosses       :   Numpy array of loss at all iterations
#What it does   :
#                   This function is used to build a linear regressor
def findLinearRegressor(X, weights, Y, learningRate, gd="batch", loss="lms", debug=False):

    w = np.zeros((X.shape[1]+1,))
    newX = np.ones((X.shape[0],X.shape[1]+1))
    newX[:,1:] = X.copy()

    w, allLosses = performGradientDescent(newX,weights,Y,w,learningRate,gd,loss,debug)

    return w, allLosses
