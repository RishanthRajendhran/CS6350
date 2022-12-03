<pre>
usage: predict.py [-h] [-ds {bankNote}] [-hmv] [-debug] [-savePlots] [-v]
                  [-nvi NUMVALINSTANCES] [-vs VALIDATIONSPLIT] [-shuffle]
                  [-r LEARNINGRATE [LEARNINGRATE ...]] [-T NUMEPOCHS]
                  [-lrScheme {constant,scheme1,scheme2}] [-d D [D ...]] [-C C]
                  [-variance VARIANCE [VARIANCE ...]] [-obj {map,ml}]

optional arguments:
  -h, --help            show this help message and exit
  -ds {bankNote}, --dataset {bankNote}
                        Data set to use for building learner
  -hmv, --handleMissingVals
                        Boolean flag indicating whether missing values need to
                        be imputed
  -debug                Boolean flag to indicate whether running in debug mode
  -savePlots            Boolean flag to indicate whether to save plots when
                        running in debug mode
  -v, --validate        Boolean flag to indicate if train-validation split
                        needs to be performed
  -nvi NUMVALINSTANCES, --numValInstances NUMVALINSTANCES
                        Number of validation instances
  -vs VALIDATIONSPLIT, --validationSplit VALIDATIONSPLIT
                        Validation Split as a fraction
  -shuffle              Boolean flag to indicate whether to shuffle the train
                        data
  -r LEARNINGRATE [LEARNINGRATE ...], --learningRate LEARNINGRATE [LEARNINGRATE ...]
                        Learning Rate for SGD
  -T NUMEPOCHS, --numEpochs NUMEPOCHS
                        Number of epochs
  -lrScheme {constant,scheme1,scheme2}
                        Scheme for learning rate decomposition
  -d D [D ...]          Value of d in learning rate decomposition scheme 1
  -C C                  Value of C in learning objective
  -variance VARIANCE [VARIANCE ...]
                        Value of variance of Gaussian distribution from which
                        weights are drawn
  -obj {map,ml}         Objective to use
</pre>