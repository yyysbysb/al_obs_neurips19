#!/usr/bin/env python3

import numpy as np
import logger
import data
import learning
import model as MModel
import bandit_learning as bl
import batch_learning_v2 as bl_v2
import config as MConfig
import utils
import experiments as MExperiments

class CPolicyParameters(object):
    pass

def set_paras_dist(pos_corr_flag, paras, config, env):
    model = MModel.CModel()
    d = env.experiment.data_dim
    model.w = np.zeros(d)
    learn = learning.CLearning(model, MExperiments.CModelParameters(0,1))
    learn.random = env.experiment.random
    l = len(env.experiment.dataset.all_data)//10
    learn, _ = bl.batch_train(learn, env.experiment.dataset.all_data[:l], l)
    w = learn.model.w
    w = w / np.sqrt(np.inner(w, w))
    dst = sorted([np.abs(np.inner(e.x, w)) for e in env.experiment.dataset.all_data[-400:]])

    if pos_corr_flag: #Certainty
        threshold = len(dst)//2
        paras.c = 0.3/(np.square(dst[-threshold]))
    else: #Uncertainty
        threshold = len(dst)//6
        paras.c = 0.4/(np.square(dst[-threshold])) 
    paras.w= w

def prepare_policy(policy_name, config, env):
    paras = CPolicyParameters()
    paras.q0 = config.data.q0
    paras.d = env.experiment.data_dim
    if policy_name == "Uncertainty":
        set_paras_dist(False, paras, config, env)
    elif policy_name == "Certainty":
        set_paras_dist(True, paras, config, env)
    return paras

def gen_policy(policy_name, policy_paras):
    if policy_name == "Uncertainty":
        return lambda x: max(policy_paras.q0, min(1,1-policy_paras.c*(np.square(np.inner(x, policy_paras.w))))),\
               policy_paras.q0
    elif policy_name == "Certainty":
        return lambda x: max(policy_paras.q0, min(1,policy_paras.c*(np.square(np.inner(x, policy_paras.w))))),\
               policy_paras.q0
    elif policy_name == "_find_logsize":
        return lambda x:1, 1

def gen_policy_by_name(policy_name, dataset, config, env):
    policy_paras = prepare_policy(policy_name, config, env)
    return gen_policy(policy_name, policy_paras)