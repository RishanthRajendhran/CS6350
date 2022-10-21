<pre>
usage: python3 predict.py [-h] [-ds {bank,creditCard,iris}] [-hmv]
                  [-wl {decisionTree}] [-nwl NUMWEAKLEARNERS] [-debug] [-v]
                  [-tune] [-nvi NUMVALINSTANCES] [-vs VALIDATIONSPLIT]
                  [-shuffle] [-sr]

optional arguments:
  -h, --help            show this help message and exit
  -ds {bank,creditCard,iris}, --dataset {bank,creditCard,iris}
                        Data set to use for building weak learner
  -hmv, --handleMissingVals
                        Boolean flag indicating whether missing values need to
                        be imputed
  -wl {decisionTree}, --weakLearner {decisionTree}
                        Name of weak learner to learn
  -nwl NUMWEAKLEARNERS, --numWeakLearners NUMWEAKLEARNERS
                        Number of weak learners to learn
  -debug                Boolean flag to indicate whether running in debug mode
  -v, --validate        Boolean flag to indicate if train-validation split
                        needs to be performed
  -tune, --tuneNumWeakLearners
                        Boolean flag to indicate if numWeakLearners tuning
                        needs to be done
  -nvi NUMVALINSTANCES, --numValInstances NUMVALINSTANCES
                        Number of validation instances
  -vs VALIDATIONSPLIT, --validationSplit VALIDATIONSPLIT
                        Validation Split as a fraction
  -shuffle              Boolean flag to indicate whether to shuffle the train
                        data
  -sr, --saveResults    Boolean flag to indicate whether to save
                        visualizations of results
</pre>