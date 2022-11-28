<pre>
usage: predict.py [-h] [-ds {bankNote}] [-hmv] [-debug] [-savePlots] [-v]
                  [-nvi NUMVALINSTANCES] [-vs VALIDATIONSPLIT] [-shuffle]
                  [-r LEARNINGRATE [LEARNINGRATE ...]]
                  [-width WIDTH [WIDTH ...]] [-T NUMEPOCHS]
                  [-lrScheme {constant,scheme1,scheme2}] [-d D [D ...]]
                  [-summary] [-wInit {zeros,gaussian}] [-tensorflow]

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
  -width WIDTH [WIDTH ...]
                        Number of neurons in the hidden layer
  -T NUMEPOCHS, --numEpochs NUMEPOCHS
                        Number of epochs
  -lrScheme {constant,scheme1,scheme2}
                        Scheme for learning rate decomposition
  -d D [D ...]          Value of d in learning rate decomposition scheme 1
  -summary              Boolean flag to indicate whether to print model
                        summary
  -wInit {zeros,gaussian}, --weightsInitialization {zeros,gaussian}
                        Scheme for initialization of weights
  -tensorflow           Boolean flag to indicate whether to use tensorflow to
                        build neural network
</pre>