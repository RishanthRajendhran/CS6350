from Modules.helper.imports.packageImports import np

#makeBootstrapSamples
#Input          :
#                   X               :   Numpy matrix of data instances
#                   weights         :   Numpy vector of weights of training
#                                       instances in X
#                   m               :   Number of samples to create
#                                       Default: Random number between 
#                                                [1,len(X)]
#                   randomForest    :   Boolean flag to switch to RandomForest 
#                                       mode
#                                       Default: False
#                   g               :   Number of attributes to select for 
#                                       RandomForest
#                                       Default: Random number between 
#                                                [1,num_of_cols(X)]
#                   replacement     :   Boolean flag indicating whether samples 
#                                       are drawn with replacement
#                                       Default: True
#Output         :
#                   drawnRows               :   Numpy array of chosen rows
#                   drawnCols               :   Numpy array of chosen cols
#What it does   :
#                   This function is used to create bootstrapped
#                   samples
def makeBootstrapSamples(X, weights, m=None, randomForest=False, g=None, replacement=True):
    if m == None:
        m = np.random.randint(1,len(X))
    drawnRows = np.random.choice(np.arange(len(X)),m,p=weights,replace=replacement)
    if randomForest:
        if g == None:
            g = np.random.randint(1,X.shape[1]+1)
        drawnCols = np.random.choice(np.arange(X.shape[1]),g,replace=False)
    else:
        drawnCols = np.arange(X.shape[1])
    return drawnRows, drawnCols
