from Modules.DecisionTree.helper.imports.packageImports import np

#makePredictions
#Input          :
#                   X               :   Numpy matrix of data instances
#                   w               :   Numpy array of initial weights of linear regressor
#                   modifyX         :   Boolean flag to indicate whether to add x_0 to X
#                                       Default: False
#Output         :
#                   _               :   Numpy array of predictions
#What it does   :
#                   This function is used to predict using a linear regressor
def makePredictions(X, w, modifyX=False):
    newX = X.copy()
    if modifyX:
        newX = np.ones((X.shape[0],X.shape[1]+1))
        newX[:,1:] = X.copy()
    return np.matmul(newX,w)
