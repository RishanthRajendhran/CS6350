from Modules.DecisionTree.helper.imports.packageImports import np

#getLoss
#Input          :
#                   preds       :   Numpy matrix of predictions
#                   weights     :   Numpy array of weights of predictions
#                   Y           :   Numpy array of target labels of instances in X
#                   w           :   Numpy array of initial weights of linear regressor
#                   loss        :   Type of loss function to use
#                                   Default: lms
#Output         :
#                   _           :   Loss of w over X
#What it does   :
#                   This function is used to calculate loss of a linear regressor
def getLoss(preds, weights, Y, w, loss="lms"):
    if loss=="lms":
        return (np.sum((np.array(Y).astype(np.float64)-np.array(preds).astype(np.float64))**2))/2
    return None
