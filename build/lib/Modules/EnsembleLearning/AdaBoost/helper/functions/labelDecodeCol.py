from Modules.EnsembleLearning.AdaBoost.helper.imports.packageImports import np

#labelDecodeCol
#Input          :
#                   col             :   Array/Array of arrays to label decode
#                   le              :   Trained LabelDecoder object
#Output         :
#                   _               :   Label decoded array/array of arrays
#What it does   :
#                   This function is used to label decode a column or list 
#                   of columns
def labelDecodeCol(col, le):
    if len(np.array(col).shape)==1:
        return le.inverse_transform(col)
    newCol = []
    for row in col:
        print(row)
        newCol.append(le.inverse_transform(row))
    return np.array(newCol)