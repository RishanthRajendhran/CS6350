from Modules.Perceptron.helper.imports.packageImports import np

#predict
#Input          :
#                   X               :   Numpy matrix of data instances
#                   model           :   Dictionary containing description of perceptron model
#Output         :
#                   _               :   Numpy array of predictions
#What it does   :
#                   This function is used to predict binary labels using a perceptron
def predict(X, model):
    if model["pa"] == "standard":
        return (np.sign(np.dot(X,model["w"])+model["b"])+1)/2
    if model["pa"] == "voted":
        allPreds = np.dot(model["Ws"], X.T) + model["Bs"][:,np.newaxis] 
        return (np.sign(np.sum((model["Cs"]*allPreds.T).T,axis=0))+1)/2
    if model["pa"] == "averaged":
        return (np.sign(np.dot(X,model["a"])+model["z"])+1)/2