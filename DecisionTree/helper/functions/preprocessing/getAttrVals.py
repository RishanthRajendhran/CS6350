from helper.imports.packageImports import np
#getAttrVals
#Input          :
#                   X       : Numpy matrix of data instances
#                   locs    : List of indices of attributes for which unique values are to be
#                             generated
#                             Default: All indices in X
#Output         :
#                   vals    : Dictionary of unique values taken by the attributes in locs indexed
#                             by the index of the attribute in X
#What it does   :
#                   This function is used to find all the unique values taken by specified set of 
#                   attributes in X
#Assumption     :
#                   It is assumed that all values that all specified attributes can take can be 
#                   inferred from the input X
def getAttrVals(X,locs=None):
    if locs == None:
        locs = np.arange(X.shape[1])
    vals = {}
    for loc in locs:
        vals[loc] = np.unique(X[:,loc])
    return vals