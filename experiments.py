#!/usr/bin/env python3

import multiprocessing as mp
import time
import math
import enum
import numpy as np
import logger as MLogger
import data as MData
import learning as MLearning
import model as MModel
import policy as MPolicy
import config as MConfig
import bandit_learning as bl
import batch_learning_v2 as bl_v2
import utils

class CDataParameters(object):
    def __init__(self, prop_train, prop_log, cnt_log, Q0, xi0):
        self.prop_train = prop_train
        self.prop_log = prop_log
        self.cnt_log = cnt_log
        self.Q0 = Q0
        self.xi0 = xi0
    def copy(self):
        return CDataParameters(self.prop_train, self.prop_log, self.cnt_log, self.Q0, self.xi0)

class COptMethod(enum.Enum):
    gd_shuffle = 0
    gd_shuffle_var = 3

class CModelParameters(object):
    def __init__(self, c0, learning_rate=0.05, init_log_prop=None, \
                batch_sz=None, batch_rate=None, label_budget=None, opt_method=COptMethod.gd_shuffle):
        self.c0 = c0
        self.learning_rate = learning_rate
        self.init_log_prop = init_log_prop
        self.batch_sz = batch_sz
        self.batch_rate = batch_rate
        self.label_budget = label_budget
        self.opt_method = opt_method
    def copy(self):
        return CModelParameters(self.c0, self.learning_rate, self.init_log_prop, self.batch_sz, \
            self.batch_rate, self.label_budget, self.opt_method)

def run_batch(algo, config, env, log):
    if algo != bl.IDBAL:
        log.set_cond_log(config.experiment.batch_sz, config.experiment.batch_rate)
        algo(config, env, log)        
    else:
        env.experiment.dataset.random = env.experiment.random
        data_paras = CDataParameters(prop_train=config.data.prop_train, prop_log=env.experiment.prop_log,\
                        cnt_log=env.experiment.cnt_log, Q0=env.policy.Q0, xi0=env.policy.xi0)
        model_paras = CModelParameters(c0=env.model.c0, learning_rate=env.model.learning_rate, \
                        init_log_prop=config.model.init_log_prop,\
                        batch_sz = config.experiment.batch_sz, batch_rate=config.experiment.batch_rate,\
                        label_budget=env.experiment.label_budget, \
                        opt_method=COptMethod[config.model.opt_method])
        dataset = env.experiment.dataset

        all_online = dataset.online_data
        dataset.online_data = []
        bl.passive_batch(dataset, log, data_paras, model_paras)
        online_sz = config.experiment.batch_sz
        while online_sz <= len(all_online):
            dataset.online_data = all_online[:online_sz]
            learning = algo(dataset, log, data_paras, model_paras)
            if model_paras.label_budget is not None and learning.cnt_labels>model_paras.label_budget*2:
                break
            if online_sz<len(all_online) and online_sz*config.experiment.batch_rate>=len(all_online):
                online_sz = len(all_online)
            else:
                online_sz = int(online_sz * config.experiment.batch_rate)

def run_once(i, config, env, log):
    #arguments have to be already copied
    env.experiment.dataset.random_split(config.data.prop_train, env.experiment.random)
    env.experiment.dataset.split_log(env.experiment.prop_log)
    env.experiment.cnt_log = len(env.experiment.dataset.log_data)
    if hasattr(env.policy, "policy_paras"):
        env.policy.Q0, env.policy.xi0 = \
            MPolicy.gen_policy(env.experiment.policy_name, env.policy.policy_paras)
    else:
        env.policy.Q0, env.policy.xi0 = \
            MPolicy.gen_policy_by_name(env.experiment.policy_name, env.experiment.dataset, \
                                       config, env)
    env.experiment.dataset.log_data = MData.gen_synthetic_bandit(env.experiment.dataset.log_data, \
                                       env.policy.Q0, env.experiment.random)
    run_batch(GAlgoDict[env.experiment.algo_name][0], config, env, log)
    return log

def run_multiple(config, env, log, num_iter):
    tmp_loggers = [MLogger.CLogger(log.info) for i in range(0, num_iter)]
    tmp_envs = [env.prepare_for_multicore(config, i) for i in range(0, num_iter)]
    if config.experiment.num_processes==1:
        for i in range(0, num_iter):    
            run_once(i, config.copy(), tmp_envs[i], tmp_loggers[i])            
    else:                 
        with mp.Pool(processes=config.experiment.num_processes) as pool:    
            res = [pool.apply_async(run_once, [i, config.copy(), tmp_envs[i], tmp_loggers[i]])\
                    for i in range(0, num_iter)]
            tmp_loggers = [r.get() for r in res]
    log.init_by_merge(tmp_loggers)

def get_max_cnt_label(dataset, prop_train, prop_log, label_budget=-1):
    len_online = int(len(dataset.all_data)*prop_train*(1-prop_log))
    if label_budget<0 or len_online<label_budget:
        return len_online
    else:
        return label_budget

def tune_batch(info, config, env, algo_name, tune_iter, log_file=None, return_all=False):
    print("start tuning for ", info)
    best_auc = 1e10

    env.experiment.algo_name = algo_name
    if not GAlgoDict[algo_name][2]:
        paras = [MConfig.CEnvironment(init_kv={"learning_rate": lr, "c0":0}) \
                 for lr in config.model.learning_rates]
    else:
        paras = [MConfig.CEnvironment(init_kv={"learning_rate": lr, "c0":c0}) \
                 for c0 in config.model.c0s for lr in config.model.learning_rates]
    for para in paras:
        logger = MLogger.CLogger(str(para.c0))
        env.model = para
        run_multiple(config, env, logger, tune_iter)
        logger.get_stat(max_cnt_label=env.experiment.label_budget)
        
        if log_file != None:
            log_info = info + " " + str(para.c0) + " " + str(para.learning_rate)
        else:
            log_info = None
        auc = MLogger.calc_metric(logger, \
                get_max_cnt_label(env.experiment.dataset, config.data.prop_train, env.experiment.prop_log, env.experiment.label_budget), \
                config.experiment.logscale_flag, log_file=log_file, log_info=log_info, \
                extra_tail=config.experiment.extra_tail)
        
        print(para.c0, " ", para.learning_rate, " ", auc)
        if auc<best_auc:
            best_auc = auc
            best_para = para
            best_log = logger

        if config.experiment.debug:
            break
    if return_all:
        return best_auc, best_para, best_log
    else:
        return best_para

GAlgoDict = {
        "idbal": (bl.IDBAL, "Active18", True),\
        "passive_is_v2": (bl_v2.passive_batch_v2, "Passive", False), \
        "active_mis_vc_debias_clipcap_v2": (bl_v2.active_MIS_vc_debias_clipcap, "ActiveVC", True), \
    }
    #Third field: has_c0

def choose_prop_log(config, env, log_file):
    if config.experiment.debug:
        num_iter = 1
    else:
        num_iter = 4

    dataset = env.experiment.dataset
    dataset_name = env.experiment.dataset_name
    policy_name = env.experiment.policy_name
    batch_sz = int(len(dataset.all_data) * 0.8 / 200)

    tune_config = config.copy()
    tune_config.experiment.batch_sz = batch_sz
    tune_config.experiment.batch_rate = 1
    tune_config.experiment.init_log_prop = 0
    tune_config.experiment.suggested_label_budget = -1
    env.experiment.label_budget = config.experiment.suggested_label_budget
    env.experiment.prop_log = 0.005
    l = tune_batch("%s %s %s"%(policy_name, dataset_name, "Choose Prop Log"), \
            config, env, "passive_is_v2", num_iter, log_file=None, return_all=True)[2]
    
    assert isinstance(l, MLogger.CLogger) 
    al_start_err = l.err_stat[0][-1] + (l.err_stat[0][1]-l.err_stat[0][-1])*0.2
    al_end_err = l.err_stat[0][-1] + (l.err_stat[0][1]-l.err_stat[0][-1])*0.05
    idx = 1
    while idx<len(l.err_stat[0]):
        if utils.mean(l.err_stat[0][idx-1: idx+2])<=al_start_err and l.err_stat[0][idx]<=al_start_err:
            break
        idx+=1
    if idx==len(l.err_stat[0]): idx -=1
    target_log_cnt = l.label_stat[0][idx]

    while idx<len(l.err_stat[0]):
        if utils.mean(l.err_stat[0][idx-1: idx+2])<=al_end_err and l.err_stat[0][idx]<=al_end_err:
            break
        idx+=1
    if idx==len(l.err_stat[0]): idx -=1
    target_al_cnt = l.label_stat[0][idx]

    r = np.random.RandomState(seed=env.experiment.random.randint(100000))
    dataset_cp = dataset.copy_all()
    dataset_cp.random = r
    dataset_cp.random_split(0.8, r)
    p, _ = MPolicy.gen_policy_by_name(policy_name, dataset_cp, config, env)  
    sz, e_cnt = 0, 0
    while sz<len(dataset_cp.train_data)*0.6:
        e_cnt += p(dataset_cp.train_data[sz].x)
        if e_cnt>target_log_cnt:
            break 
        sz+=1
    prop_log = max(0.1, sz/len(dataset_cp.train_data))
    label_budget=max((target_al_cnt-target_log_cnt)//10, 100)
    if log_file is not None:
        log_file.write("prop_log: [(%.2f, %.2f, %.2f, %.2f), (%.3f, %.3f)]\n"\
            %(prop_log, sz, target_log_cnt, label_budget, al_start_err, al_end_err))
    print("prop_log: %.2f\tcnt_log: %2f\teffective_log: %.2f\tbudget: %.2f\terr: %.3f, %.3f"\
            %(prop_log, sz, e_cnt, label_budget, al_start_err, al_end_err))
    return prop_log, label_budget

def run_experiments(config, env, auclogger=None, medianauc_logger=None, log_file=None):
    if env.experiment.dataset is None:
        return
    dataset = env.experiment.dataset
    dataset_name = env.experiment.dataset_name
    policy_name = env.experiment.policy_name
    output_filename = config.experiment.result_path+dataset_name+"-"+policy_name
    tune_iter = 1 if config.experiment.debug else config.experiment.tune_iter 
    final_iter = 1 if config.experiment.debug else config.experiment.final_iter
    print("[%s] Experiment for %s"%(time.asctime(), output_filename))

    if config.experiment.dynamic_log_size:
        prop_log, _label_budget = choose_prop_log(config, env, log_file)
        if config.experiment.suggested_label_budget==-1:
            env.experiment.label_budget = _label_budget
        else:
            env.experiment.label_budget = config.experiment.suggested_label_budget
        env.experiment.prop_log = prop_log
    else:
        env.experiment.label_budget = config.experiment.suggested_label_budget
        env.experiment.prop_log = config.data.suggested_prop_log

    if log_file != None:
        log_file.write("prop_log: %.2f, label_budget: %d\n"%(env.experiment.prop_log, env.experiment.label_budget))

    env.policy.policy_paras = MPolicy.prepare_policy(policy_name, config, env)
        
    results = [(a, tune_batch("%s %s %s"%(policy_name, dataset_name, a), \
                                               config, env, a, tune_iter, log_file, return_all=True)) \
             for a in config.experiment.algos]

    if log_file != None:
        log_file.write("\nBEST:\n")
    for group in results:
        algo_name = group[0]
        auc = group[1][0]
        paras = group[1][1]
        logger = group[1][2]

        medianauc = MLogger.calc_median_metric(logger, \
                get_max_cnt_label(dataset, config.data.prop_train, env.experiment.prop_log, env.experiment.label_budget), \
                config.experiment.logscale_flag, extra_tail=config.experiment.extra_tail)
        logger.info = algo_name

        print(algo_name, ": ", paras.c0, " ", paras.learning_rate, "\t\t", auc, "\t", medianauc)
        if auclogger is not None:
            auclogger.log(policy_name, dataset_name, algo_name, auc)
        if medianauc_logger is not None:
            medianauc_logger.log(policy_name, dataset_name, algo_name, medianauc)

        if log_file != None:
            MLogger.log_metric(logger, auc, log_file, \
                policy_name + " " + dataset_name + " " + algo_name + " " + str(paras.c0) \
                + " " + str(paras.learning_rate))
        

    MLogger.plot_err([group[1][2] for group in results], False, output_filename)
    MLogger.plot_err([group[1][2] for group in results], True, output_filename)

    if log_file != None:
        log_file.write("===========================================\n\n")

