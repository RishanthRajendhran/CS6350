from Modules.DecisionTree.helper.imports.packageImports import np
#getGiniIndex
#Input          :
#                   ns : A numerical type list of counts of all class labels
#Output         :
#                   Gini Index 
# What it does  :
#                   This function computes the gini index of a given data set given
#                   the numerical counts of all class labels in that data set
def getGiniIndex(ns):
    if len(ns)==0:
        return 0
    N = np.sum(ns)
    gini = 1 - np.sum((np.array(ns)/N)**2)
    return gini