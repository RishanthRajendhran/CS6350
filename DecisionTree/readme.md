The predict.py file supports making predictions on two data sets:
1. Car
2. Bank
The program processes the input data (e.g. converts categorical attribures into numerical values) and then builds a decision tree.
The program can be run as follows:
"python3 predict.py car" for the Car data set
"python3 predict.py bank" for the Bank data set
The program when excecuted prints the training and test data errors for different information gain criteria (specifically: entropy, majority error, gini index)
The maximum tree depths are hardcoded in the helper/config/dataConfig.py
Other (hyper)parameters such as data type of labels, numerical attributes, categorical attributes etc are also defined in helper/config/dataConfig.py
