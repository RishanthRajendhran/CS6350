from Modules.helper.imports.packageImports import np

allowedLearners = ["decisionTree"]
learner = "decisionTree"
fracBootstrapSamples = 0.5
m = None
numLearners = 100
validationSplit = 0.3
shuffleData = False
conditions = {
    "maxDepth": np.inf,
    "metric": "e",
    "replacement":True,
}
randomForest = False
fracAttrs = 0.5
g = None
numAttrsList = []