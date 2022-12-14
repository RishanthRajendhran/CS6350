from Modules.DecisionTree.helper.imports.packageImports import np
from Modules.DecisionTree.helper.imports.configImports import miscConfig
#getGain
#Input          :
#                   ns          :   A numerical type list of list of counts of all class labels
#                   numClasses  :   Number of class labels
#                   metric      :   Metric to calculate gain with 
#Output         :
#                   _           :   Gain rounded to 4 decimal places
#What it does   :
#                   This function computes the gain of a split of a given data set over an attribute given
#                   the probabilities of all class labels in the original and split data sets
def getGain(ns, numClasses=2, metric="e"):  
    vals = [miscConfig.metricsMap[metric](np.array(ns[i])) for i in range(len(ns))]
    gain = vals[0]
    for i in range(1,len(vals)):
        gain -= (np.sum(np.array(ns[i]))/np.sum(np.array(ns[0])))*vals[i]
    return gain