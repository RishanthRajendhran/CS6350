<pre>
usage: predict.py [-h] [-ds {bankNote}] [-hmv] [-debug] [-savePlots] [-v]
                  [-nvi NUMVALINSTANCES] [-vs VALIDATIONSPLIT] [-shuffle]
                  [-r LEARNINGRATE [LEARNINGRATE ...]]
                  [-gamma GAMMA [GAMMA ...]] [-T NUMEPOCHS]
                  [-lrScheme {constant,scheme1,scheme2}] [-a A [A ...]]
                  [-C C [C ...]] [-form {primal,dual}] [-kernel {gaussian}]

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
  -gamma GAMMA [GAMMA ...]
                        Gamma for Gaussian Kernel for dual form of SVM
  -T NUMEPOCHS, --numEpochs NUMEPOCHS
                        Number of epochs
  -lrScheme {constant,scheme1,scheme2}
                        Scheme for learning rate decomposition
  -a A [A ...]          Value of a in learning rate decomposition scheme 1
  -C C [C ...]          Value of C in SVM objective
  -form {primal,dual}   Form of SVM objective function to use
  -kernel {gaussian}    Type of kernel to use for dual form of objective
</pre>