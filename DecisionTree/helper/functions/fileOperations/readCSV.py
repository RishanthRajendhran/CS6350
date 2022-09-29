from helper.imports.packageImports import np, csv
#readCSV
#Input          :
#                   fileName      : Path to CSV file to be read
#                   skipHeader    : Boolean variable indicating whether to skip the header
#                                   in the CSV file or not
#                                   Default: True
#Output         :
#                   out        : Numpy matrix with rows as the rows of the CSV file and columns
#                                as the value between commas in each row
#What it does   :
#                   This function is used to read CSV files
def readCSV(fileName, skipHeader=True):
    with open(fileName, "r") as csvFile: 
        out = np.array(list(csv.reader(csvFile, delimiter=",")))
    if skipHeader:
        out = out[1:,:]
    return out