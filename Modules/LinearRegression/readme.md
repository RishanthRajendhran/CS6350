<pre>
usage: python3 predict.py [-h] [-ds {concrete}] [-hmv] [-debug] [-v]
                  [-nvi NUMVALINSTANCES] [-vs VALIDATIONSPLIT] [-shuffle]
                  [-gd {batch,stochastic}] [-l {lms}]
                  [-lr LEARNINGRATE [LEARNINGRATE ...]] [-sr]

optional arguments:
  -h, --help            show this help message and exit
  -ds {concrete}, --dataset {concrete}
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
  -gd {batch,stochastic}, --gradientDescent {batch,stochastic}
                        Type of gradient descent to perform
  -l {lms}, --loss {lms}
                        Type of loss to use
  -lr LEARNINGRATE [LEARNINGRATE ...], --learningRate LEARNINGRATE [LEARNINGRATE ...]
                        Learning Rate for GD
  -sr, --saveResults    Boolean flag to indicate whether to save
                        visualizations of results
</pre>