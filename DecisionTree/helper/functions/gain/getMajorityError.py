from helper.imports.packageImports import np
#getMajorityError
#Input          :
#                   ns : A numerical type list of counts of all class labels
#Output         :
#                   Majority Error rounded to 4 decimal places
# What it does  :
#                   This function computes the majority error of a given data set given
#                   the numerical counts of all class labels in that data set
def getMajorityError(ns):
    if len(ns)==0:
        return 0
    N = np.sum(ns)
    me = (N-max(ns))/N
    return np.round(me,4)