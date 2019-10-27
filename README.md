# The Label Complexity of Active Learning from Observational Data
This repository contains a python implementation of experiments in paper:
> Songbai Yan, Kamalika Chaudhuri, Tara Javidi. "The Label Complexity of Active Learning from Observational Data." Advances in Neural Information Processing Systems. 2019. 

## File Descriptions
- bandit_learning.py Implementation of [1]. Copied from https://github.com/yyysbysb/al_log_icml18
- batch_learning_v2.py Implementation of the proposed algorithm
- config.py Setting up the configuration of experiments
- data.py   Preprocessing data
- experiments.py    Implementation of experiments
- learning.py   Updating the model given a set of data
- logger.py Logging information for plots and reports
- main.py   The main file
- model.py  Implementation of a linear model
- opt.py    Optimization methods
- policy.py Setting up logging policies
- utils.py  Auxiliary functions
- config.json   The configuration of experiments

## External libraries
- numpy

## References
1. Songbai Yan, Kamalika Chaudhuri, Tara Javidi. "Active Learning with Logged Data." Proceedings of the 35th International Conference on Machine Learning, PMLR 80:5521-5530, 2018.