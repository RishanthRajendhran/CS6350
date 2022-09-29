from helper.imports.packageImports import np
from helper.imports.configImports import miscConfig
#getGain
#Input          :
#                   ns          :   A numerical type list of counts of all class labels
#                   numClasses  :   Number of class labels
#                   metric      :   Metric to calculate gain with 
#Output         :
#                   _           :   Gain rounded to 4 decimal places
# What it does  :
#                   This function computes the gain of a split of a given data set over an attribute given
#                   the probabilities of all class labels in the original and split data sets
def getGain(ns, numClasses=2, metric="e"):  
    if len(ns)%numClasses:
        print("Unexpected size for ns")
        exit(0)
    vals = [miscConfig.metricsMap[metric](np.array(ns)[i*numClasses:i*numClasses+numClasses]) for i in range(len(ns)//numClasses)]
    gain = vals[0]
    for i in range(1,len(vals)):
        gain -= (np.sum(np.array(ns)[i*numClasses:i*numClasses+numClasses])/np.sum(np.array(ns[0:numClasses])))*vals[i]
    return np.round(gain,4)