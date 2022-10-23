from Modules.EnsembleLearning.AdaBoost.helper.imports.packageImports import np

#labelEncodeCol
#Input          :
#                   col             :   Array/Array of arrays to label encode
#                   le              :   Trained LabelEncoder object
#Output         :
#                   _               :   Label encoded array/array of arrays
#What it does   :
#                   This function is used to label encode a column or list 
#                   of columns
def labelEncodeCol(col, le):
    if len(np.array(col).shape)==1:
        return le.transform(col)
    newCol = []
    for row in col:
        newCol.append(le.transform(row))
    return np.array(newCol)