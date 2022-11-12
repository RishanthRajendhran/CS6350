<pre>
usage: predict.py [-h] [-ds {bankNote}] [-hmv] [-debug] [-v]
                  [-nvi NUMVALINSTANCES] [-vs VALIDATIONSPLIT] [-shuffle]
                  [-pa {standard,voted,averaged,guassianKernel}]
                  [-r LEARNINGRATE [LEARNINGRATE ...]] [-T NUMEPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -ds {bankNote}, --dataset {bankNote}
                        Data set to use for building learner
  -hmv, --handleMissingVals
                        Boolean flag indicating whether missing values need to
                        be imputed
  -debug                Boolean flag to indicate whether running in debug mode
  -v, --validate        Boolean flag to indicate if train-validation split
                        needs to be performed
  -nvi NUMVALINSTANCES, --numValInstances NUMVALINSTANCES
                        Number of validation instances
  -vs VALIDATIONSPLIT, --validationSplit VALIDATIONSPLIT
                        Validation Split as a fraction
  -shuffle              Boolean flag to indicate whether to shuffle the train
                        data
  -pa {standard,voted,averaged,guassianKernel}, --perceptronAlgorithm {standard,voted,averaged,gaussianKernel}
                        Perceptron algorithm to use
  -r LEARNINGRATE [LEARNINGRATE ...], --learningRate LEARNINGRATE [LEARNINGRATE ...]
                        Learning Rate for GD
  -T NUMEPOCHS, --numEpochs NUMEPOCHS
                        Number of epochs
</pre>