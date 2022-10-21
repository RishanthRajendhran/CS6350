import Modules.Results
from Modules.DecisionTree.helper.imports.packageImports import np, plt

#visualizeResults
#Input          :
#                   results         :   Numpy matrix of results to be visualized
#                   saveResults     :   Boolean flag to indicate whether to save 
#                                       visualizations
#                                       Default: False
#                   fileName        :   File name for saving results
#                                       Default: results.png
#                   clearPlot       :   Boolean flag to indicate whether to clear 
#                                       the generated plot before returning
#                                       Default: True
#Output         :
#                   None
#What it does   :
#                   This function is used to visualize results 
def visualizeResults(results, saveResults=False, fileName="results", clearPlot=True):
    plt.plot(results)  
    plt.xlabel("Iterations") 
    plt.ylabel(fileName) 
    if saveResults and clearPlot:
        plt.savefig(f"{str(list(Modules.Results.__path__)[0])}/{fileName}.png")
    elif clearPlot:
        plt.show()
    if clearPlot:
        plt.clf()