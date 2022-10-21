from Modules.helper.imports.packageImports import np, argparse
from Modules.helper.functions.fileOperations.readCSV import readCSV
from Modules.helper.functions.fileOperations.trainValSplit import trainValSplit

parser = argparse.ArgumentParser()

parser.add_argument(
    "-train",
    "--train",
    help="Path to trainX",
    required=True,
)

parser.add_argument(
    "-test",
    "--testDir",
    help="Path to directory to save test files",
    required=True,
)

parser.add_argument(
    "-sh",
    "--skipHeader",
    action="store_true",
    help="Boolean flag to indicate whether to skip header",
)

parser.add_argument(
    "-nhl",
    "--numHeaderLines",
    type=int, 
    help="Number of lines in the header",
)

parser.add_argument(
    "-nti",
    "--numTestInstances",
    type=int, 
    help="Number of test instances",
)

parser.add_argument(
    "-ts",
    "--testSplit",
    type=float, 
    help="Test Split as a fraction",
)

parser.add_argument(
    "-shuffle",
    action="store_true",
    help="Boolean flag to indicate whether to shuffle the train data",
)

args = parser.parse_args()
trainPath = args.train
testDir = args.testDir
skipHeader = args.skipHeader
numHeaderLines = 0
testSplit = 0.3
shuffle = args.shuffle
if args.testSplit:
    testSplit = args.testSplit
if args.numHeaderLines:
    numHeaderLines = args.numHeaderLines

train = readCSV(trainPath, skipHeader, numHeaderLines)

train_X = train[:,1:-1]
train_Y = train[:,-1]

weights = np.ones((len(train_X),))
weights /= len(weights)

if args.numTestInstances and len(train_X) >= args.numTestInstances:
    testSplit = args.numTestInstances/len(train_X)

(train_X, _, train_Y, test_X, _, test_Y) = trainValSplit(train_X, weights, train_Y, testSplit, shuffle)

train_X = np.concatenate((train_X, train_Y.reshape(len(train_Y),-1)), axis=1)
np.savetxt(f"{testDir}/trainX.csv",train_X, delimiter=",",fmt="%s")

test_X = np.concatenate((test_X, test_Y.reshape(len(test_Y),-1)), axis=1)
np.savetxt(f"{testDir}/testX.csv",test_X, delimiter=",",fmt="%s")


