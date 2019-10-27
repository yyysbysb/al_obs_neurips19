#!/usr/bin/env python3

import os
import sys
import multiprocessing as mp
import numpy as np
import experiments
import data
import logger as MLogger
import config as MConfig
import utils

def backup_code(config):
    result_folder = config.experiment.result_path
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    utils.backup_code(result_folder)

def run_experiments(config):
    result_folder = config.experiment.result_path
    auc_logger = MLogger.CAUCLogger(order=config.experiment.dataset_set)
    auc_logger.write(result_folder+"auc.res", append=False)
    medianauc_logger = MLogger.CAUCLogger(order=config.experiment.dataset_set)
    medianauc_logger.write(result_folder+"auc_median.res", append=False)

    with open(result_folder+"metric.log", "w") as log_file:
        for policy_name in config.experiment.policy_set:
            for dataset_name in config.experiment.dataset_set:
                env = MConfig.CEnvironment(config=config)
                env.experiment.random = np.random.RandomState(config.experiment.rand_seed)
                env.experiment.policy_name = policy_name
                env.experiment.dataset_name = dataset_name
                env.experiment.dataset = data.load_data(dataset_name, env.experiment.random, config.data.max_sz)
                env.experiment.data_dim = env.experiment.dataset.all_data[0].x.shape[0]
                
                #todo: remove following fields
                env.dataset = env.experiment.dataset
                config.example_dim = env.experiment.data_dim
                config.q0 = config.data.q0

                experiments.run_experiments(config, env,\
                    auclogger=auc_logger, medianauc_logger=medianauc_logger, log_file=log_file)    
            auc_logger.write(result_folder+"auc.res", append=True)
            auc_logger.clear()
            medianauc_logger.write(result_folder+"auc_median.res", append=True)
            medianauc_logger.clear()

def main():
    print(os.getcwd())
    sys.stdout.flush()
    if (len(sys.argv)>1):
        print(sys.argv[1])
        CONFIG_PATHS = ["./"]
        CONFIG_FILENAME = sys.argv[1]
        print("Config File = "+CONFIG_FILENAME)
    else:
        CONFIG_PATHS = ["./", "../"]
        CONFIG_FILENAME = "config.json"
        print("Using default Config file")
    
    config = MConfig.load_config(CONFIG_PATHS, CONFIG_FILENAME)
    config.experiment.result_path = config.experiment.result_root_path \
                                    + config.experiment.experiment_name \
                                    + "/"
    print(config.experiment.result_path)
    backup_code(config)
    
    run_experiments(config)

if __name__ == "__main__":
    main()
