#!/usr/bin/env python3

import numpy as np
import logger as MLogger
import data as MData
import model as MModel
import opt as MOpt
import learning as MLearning
import experiments
import config as MConfig

def batch_train(learning, data, config, env, idx=0):
    assert isinstance(learning, MLearning.CLearning)
    if len(data)==0:
        return 0

    if idx==0:
        learning.reset(config, env)
        env.idx = 0
    else:
        idx = env.idx
    if config.model.opt_shuffle:        
        data = [MData.CExample(e.x, e.y, e.w, env.experiment.random.randint(idx, idx+len(data))) for e in data if e.z>0]
        data = sorted(data, key=lambda e: e.z)
    
    ws = [np.copy(learning.model.w)]
    env.acc_var_err = []
    for i in range(0,3):
        sum_loss = learning.update_model(data, config, env, env.idx+len(data)*i)
        ws.append(np.copy(learning.model.w))
    sum_loss = sum([e.w for e in data if e.z>0 and e.w<=env.clip_th and learning.model.predict(e.x)*e.y<=0])
    sum_var = sum([e.w*e.w for e in data if e.z>0 and e.w<=env.clip_th and learning.model.predict(e.x)*e.y<=0])
    env.sum_var = sum_var

    env.idx += len(data)*3

    diff_ws_info = ["%.1E"%np.linalg.norm(ws[i]-ws[i+1]) for i in range(0, len(ws)-1)]
    clip_info = "()" if env.clip_th>1/env.policy.xi0 \
                else " (%.2E, %.2f)"%(1/env.clip_th, MLearning.calc_clip_percentage(data, env.clip_th))
    var_err_info = "()"
    if (len(env.acc_var_err)>10):
        err = sorted(env.acc_var_err)
        lerr = len(err)
        p1, p2, p3 = lerr//10, lerr//2, lerr//10*9
        var_err_info = "((%.2f, %.1E), (%.2f, %.1E), (%.2f, %.1E))"%(err[p1][0], err[p1][1], \
                        err[p2][0], err[p2][1], err[p3][0], err[p3][1])
    misc_info = "[(%s), %s, %s]"%(",".join(diff_ws_info), clip_info, var_err_info)
    env.logger.log_misc_info(misc_info)

    return sum_loss

def generic_batch_learning_v2(dataset, logger, config, env):
    assert isinstance(dataset, MData.CDataSet)
    assert isinstance(logger, MLogger.CLogger)

    learning = env.learning
    sum_loss = env.batch_train(learning, dataset.log_data, config, env)
    data_batches = [dataset.log_data]
    cur_online = 0
    cur_sz = config.experiment.batch_sz
    env.total_size = env.experiment.cnt_log

    logger.on_start(learning, dataset)
    logger.check_and_log(learning, dataset, cur_online)
    while cur_online < len(dataset.online_data)\
    and (env.experiment.label_budget==-1 or learning.cnt_labels<=env.experiment.label_budget*2):
        next_batch = [e for e in dataset.online_data[cur_online:cur_online+cur_sz]]
        cur_online += len(next_batch)
        data_batches.append(next_batch)
        env.total_size += len(next_batch)
        cur_dataset, next_batch = env.digest(learning, data_batches, sum_loss, config, env)
        data_batches[-1] = next_batch
        sum_loss = env.batch_train(learning, cur_dataset, config, env, env.experiment.cnt_log+cur_online)
        cur_sz = int(cur_sz*config.experiment.batch_rate)
        logger.check_and_log(learning, dataset, cur_online)

    logger.on_stop(learning, dataset)
    return learning

def passive_digest(learning, data_batches, sum_loss, config, env):
    learning.cnt_labels += len(data_batches[-1])
    return [e for b in data_batches for e in b], data_batches[-1]

def MIS_transform(batches, config, env, Qks=None):
    n = [len(b) for b in batches]
    n[0] = env.experiment.cnt_log
    if Qks is None:
        s = sum(n[1:])
        return [MData.CExample(e.x, e.y, (n[0]+s)/(n[0]*env.policy.Q0(e.x)+s), e.z) \
            for b in batches for e in b]
    else:
        s = sum(n)
        return [MData.CExample(e.x, e.y, s/sum([nq[0]*nq[1](e.x) for nq in zip(n, Qks)]), e.z) \
            for b in batches for e in b]

def test_dis(e, learning, sum_loss, c0, t, Mk, eta):
    if learning.cnt_labels<3:
        return True
    gap = MOpt.calc_gap(learning.model, e, t, eta)
    threshold = np.sqrt(c0*Mk*sum_loss/t/t) + c0*Mk*np.log(t)/t
    return gap/t<=threshold

def test_dis_var(e, learning, sum_var, c0, t, clip_th, eta):
    if learning.cnt_labels<3:
        return True
    gap = MOpt.calc_gap(learning.model, e, t, eta)
    threshold = np.sqrt(c0*sum_var/t/t) + c0*clip_th*np.log(t)/t
    return gap/t<=threshold

def active_MIS_clip_capped_debias_digest(learning, data_batches, sum_loss, config, env):
    m = env.experiment.cnt_log
    n = env.total_size-m
    new_batch = []
    qk = lambda x: 1 if env.policy.Q0(x)<2*n/m else 0
    Mk = 2*(m+n)/(m*env.policy.xi0+n)

    for e in data_batches[-1]:
        e_cp = MData.CExample(e.x, e.y, e.w, e.z)
        if qk(e_cp.x)>0:
            e_cp.z = 1
            e_cp.w = 1
            if test_dis_var(e_cp, learning, env.sum_var, env.model.c0, env.total_size, env.clip_th, env.model.learning_rate):
                learning.cnt_labels += 1
            else:
                e_cp.y = learning.model.predict(e_cp.x)
        else:
            e_cp.z = 0
        new_batch.append(e_cp)
    if learning.cnt_labels<10 or Mk<10:
        env.clip_th = Mk
    else:
        env.clip_th = MLearning.search_clip_threshold(env.qs, (m+n)/m, -n/m, env.model.c0/(m+n), 0)
    env.Qks.append(qk)
    data_batches[-1] = new_batch
    return MIS_transform(data_batches, config, env, env.Qks), data_batches[-1]

def prepare(config, env, logger, learning_class):    
    env.batch_train = batch_train
    env.clip_th =  10/env.policy.xi0
    env.logger = logger

    model = MModel.CModel()
    model.w = np.zeros(env.experiment.data_dim)
    env.learning = learning_class(model, config, MLearning.square_loss_val, MLearning.square_loss_grad)

def passive_batch_v2(config, env, logger):
    prepare(config, env, logger, MLearning.CLearning_v2_linear)
    env.digest = passive_digest
    env.opt = MOpt.gd_v2
    return generic_batch_learning_v2(env.experiment.dataset, logger, config, env)

def active_MIS_vc_debias_clipcap(config, env, logger):
    prepare(config, env, logger, MLearning.CLearning_v2_linear_vc)
    env.digest = active_MIS_clip_capped_debias_digest
    env.opt = MOpt.gd_var_v2
    if len(env.experiment.dataset.log_data)>200:
        env.qs = sorted([1/e.w for e in env.experiment.dataset.log_data])
    else:
        env.qs = sorted([env.policy.Q0(e.x) for e in env.experiment.dataset.train_data])    
    env.Qks = [env.policy.Q0]
    return generic_batch_learning_v2(env.experiment.dataset, logger, config, env)