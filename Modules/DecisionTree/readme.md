<pre>
usage: python3 predict.py [-h] [-ds {car,bank,creditCard}] [-hmv]
                  [-md MAXDEPTH [MAXDEPTH ...]]

optional arguments:
  -h, --help            show this help message and exit
  -ds {car,bank,creditCard}, --dataset {car,bank,creditCard}
                        Data set to use for building decision tree
  -hmv, --handleMissingVals
                        Boolean flag indicating whether missing values need to
                        be imputed
  -md MAXDEPTH [MAXDEPTH ...], --maxDepth MAXDEPTH [MAXDEPTH ...]
                        Maximum allowed depth(s) for the decision tree;
                        multiple values should be separated by a whitespace
</pre>
<p>
    The predict.py file supports making predictions on two data sets:
    <br/>
    <ol>
        <li>
            Car
        </li>
        <li>
            Bank
        </li>
    </ol>
    The program processes the input data (e.g. converts categorical attribures into numerical values) and then builds a decision tree.
    <br/>
    The program can be run as follows:
    <br/>
    "python3 predict.py -dataset car" for the Car data set
    <br/>
    "python3 predict.py -dataset bank" for the Bank data set
    <br/>
    The program when excecuted prints the training and test data errors for different information gain criteria (specifically: entropy, majority error, gini index)
    <br/>
    The maximum tree depths are hardcoded in the helper/config/dataConfig.py
    <br/>
    Other (hyper)parameters such as data type of labels, numerical attributes, categorical attributes etc are also defined in helper/config/dataConfig.py
    <br>
    The "-handleMissingVals" flag should be used to handle missing attribute values in input data matrix.
    <br/>
    The "-maxDepth" flag can be used to set allowed maximum tree depths
</p>
