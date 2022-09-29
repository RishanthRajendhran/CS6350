#makePredictions
#Input          :
#                   root    :   Root node of the decision tree
#                   X       :   Numpy matrix of data instances
#Output         :
#                   preds   : List of predictions made by the decision tree for
#                             data instances in X
#What it does   :
#                   This function is used to build a decision tree recursively
def makePredictions(root, X):
    preds = []
    for x in X:
        preds.append(root.predict(x))
    return preds