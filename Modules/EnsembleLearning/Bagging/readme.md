<pre>
usage: python3 predict.py [-h] [-ds {bank,creditCard,iris}] [-hmv] [-l {decisionTree}]
                  [-nl NUMLEARNERS] [-debug] [-v] [-tune]
                  [-nvi NUMVALINSTANCES] [-vs VALIDATIONSPLIT] [-shuffle]
                  [-sr] [-fbs FRACBOOTSTRAPSAMPLES] [-m NUMBOOTSTRAPSAMPLES]
                  [-rf] [-fs FRACATTRS] [-g NUMATTRS]
                  [-nal NUMATTRSLIST [NUMATTRSLIST ...]] [-exp]

optional arguments:
  -h, --help            show this help message and exit
  -ds {bank,creditCard,iris}, --dataset {bank,creditCard,iris}
                        Data set to use for building learner
  -hmv, --handleMissingVals
                        Boolean flag indicating whether missing values need to
                        be imputed
  -l {decisionTree}, --learner {decisionTree}
                        Name of learner to learn
  -nl NUMLEARNERS, --numLearners NUMLEARNERS
                        Number of learners to learn
  -debug                Boolean flag to indicate whether running in debug mode
  -v, --validate        Boolean flag to indicate if train-validation split
                        needs to be performed
  -tune, --tuneNumLearners
                        Boolean flag to indicate if numLearners tuning needs
                        to be done
  -nvi NUMVALINSTANCES, --numValInstances NUMVALINSTANCES
                        Number of validation instances
  -vs VALIDATIONSPLIT, --validationSplit VALIDATIONSPLIT
                        Validation Split as a fraction
  -shuffle              Boolean flag to indicate whether to shuffle the train
                        data
  -sr, --saveResults    Boolean flag to indicate whether to save
                        visualizations of results
  -fbs FRACBOOTSTRAPSAMPLES, --fracBootstrapSamples FRACBOOTSTRAPSAMPLES
                        Fraction (>0 and <=1) of data instances to use as
                        bootstrap samples for bagging
  -m NUMBOOTSTRAPSAMPLES, --numBootstrapSamples NUMBOOTSTRAPSAMPLES
                        Number of data instances to use as bootstrap samples
                        for bagging
  -rf, --randomForest   Boolean flag to indicate whether to grow a random
                        forest
  -fs FRACATTRS, --fracAttrs FRACATTRS
                        Fraction (>0 and <=1) of attributes to choose for
                        Random Forest
  -g NUMATTRS, --numAttrs NUMATTRS
                        Number of attributes to choose for Random Forest
  -nal NUMATTRSLIST [NUMATTRSLIST ...], --numAttrsList NUMATTRSLIST [NUMATTRSLIST ...]
                        List of number of attributes to pick for Random Forest
  -exp, --experiment    Boolean flag to indicate whether to switch to
                        experiment mode
</pre>