from helper.imports.packageImports import np
#handleMissingAttrValues
#Input          :
#                   X                   : Numpy matrix of data instances
#                   categoricalAttrs    : List of indices of categorical attributes in X
#                   attrVals            : Dictionary of list of unique values taken by attributes 
#                                       in X indexed by the attribute index in X
#                   modify              : Boolean variable indicating whether attrVals needs to be 
#                                         updated   
#                                         Default: True
#Output         :
#                   X                   : Numpy matrix of preprocessed data instances
#                   attrVals            : Dictionary of list of unique values taken by attributes 
#                                         in X indexed by the attribute index in X
#What it does   :
#                   This function is used to replace missing categorical attribute values with the 
#                   most frequent value for that attribute in X
#Assumption     :
#                   A value of "unknown" for an attribute in a data instance implies that the value for
#                   that attribute is missing in X
def handleMissingAttrValues(X, categoricalAttrs, attrVals, modify=True):
    for attr in categoricalAttrs:
        if "unknown" in X[:,attr]:
            vals, counts = np.unique(X[:,attr],return_counts=True)
            vals = vals.tolist()
            counts = counts.tolist()
            if "unknown" in vals:
                ind = vals.index("unknown")
                vals.remove("unknown")
                counts.remove(counts[ind])
            X[np.where(X[:,attr] == "unknown")[0],attr] = vals[np.argmax(counts)]
            if modify:
                attrVals[attr].remove("unknown")
    return X, attrVals