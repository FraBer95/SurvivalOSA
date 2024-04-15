# SurvivalOSA

This is the private repository of the "_Enhancing Survival Analysis Model Selection Through
XAI(t) in Healthcare_" paper. 
It is currently made private for review purposes but it will be made public after acceptance of the paper.

The code is written in R and Python and it is composed by two main parts:

1. Preprocessing part: Data Reading and formatting, dropping of non relevant features, categorization, statistical analysis and feature selection.
2. Analysis part: Survival Curves, Models Training and Validation, XAI(t).

The python libraries requested are: 
- _pandas, torch, pycox, reticulate, numpy, seaborn, scipy, statsmodels, scikit-learn_
  
The R libraries requested are:
- _survival, surviminer, survex, randomForestSRC, gbm, survivalsvm, ggsurvfit, ggplot2, pec, caret, SurvMetrics, mlr3proba, mlr3extralearners, mlr3pipelines, mlr3tuning, survivalmodels, reticulate_

For running CoxTime model it is suggested to install  pandas==1.5.3
