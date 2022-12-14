from Modules.DecisionTree.helper.imports.packageImports import np
#getEntropy
#Input          :
#                   ns : A numerical type list of counts of all class labels
#Output         :
#                   Entropy
# What it does  :
#                   This function computes the entropy of a given data set given
#                   the numerical counts of all class labels in that data set
def getEntropy(ns):
    if len(ns)==0:
        return 0
    ent = 0
    N = np.sum(ns)
    if N == 0:
        return 0
    for i in range(len(ns)):
        p = ns[i]/N
        if p:
            ent -= p*np.log2(p)
    return ent