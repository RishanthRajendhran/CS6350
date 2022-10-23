from Modules.DecisionTree.helper.imports.functionImports import getEntropy, getMajorityError, getGiniIndex
handleMissingVals = False
metricsMap = {
    "e": getEntropy,
    "m": getMajorityError,
    "g": getGiniIndex,
}