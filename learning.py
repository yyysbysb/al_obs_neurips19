#!/usr/bin/env python3
import numpy as np
import bisect

def square_loss_val(model, e):
    return np.power(e.y-model.predict(e.x), 2)
def square_loss_grad(model, e):
    return 2*(model.predict(e.x)-e.y)*e.x
    
class CLearning(object):
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters.copy()
        self.cnt_labels = 0

class CLearning_v2_linear(CLearning):
    def __init__(self, model, config, loss_val, loss_grad):
        CLearning.__init__(self, model, config)
        self.loss_val = loss_val # (model, e) -> double
        self.loss_grad = loss_grad # (model, e) -> vector

    def update_model(self, data, config, env, idx=0):
        sum_loss = 0
        for example in data:        
            if example.z==0:
                continue            
            else:
                idx += example.w
            if self.model.predict(example.x)*example.y<=0:
                sum_loss += example.w
            env.opt(self, example, idx, config, env)
        return sum_loss
    def reset(self, config, env):
        self.model.w = np.zeros(env.experiment.data_dim)

class CLearning_v2_linear_vc(CLearning):
    def __init__(self, model, config, loss_val, loss_grad):
        CLearning.__init__(self, model, config)
        self.loss_val = loss_val # (model, e) -> double
        self.loss_grad = loss_grad # (model, e) -> vector
        self.acc_var = 0
    def get_acc_var(self, data, config, env, subsample_size):
        if subsample_size<len(data):
            subsample = [data[env.experiment.random.randint(len(data))] for i in range(0, subsample_size)]
        else:
            subsample = data
        ratio = len(data)/len(subsample)
        return ratio*sum([np.power(e.w*self.loss_val(self.model, e), 2) \
                    for e in subsample if e.z>0 and e.w<=env.clip_th])

    def update_model(self, data, config, env, idx=0):
        sum_loss = 0        
        var_clock = 0
        env.data_size = len(data)
                    
        for example in data:  
            if example.z==0 or example.w>env.clip_th:
                continue            
            else:
                idx += example.w
                var_clock -= 1
            if var_clock<=0:
                var_diff = -self.acc_var
                self.acc_var = 1e-6+self.get_acc_var(data, config, env, config.model.opt_var_subsample_size)
                var_clock = config.model.opt_var_iter_per_update

                var_diff += self.acc_var
                env.acc_var_err.append((abs(var_diff)/self.acc_var, abs(var_diff)))

            if self.model.predict(example.x)*example.y<=0:
                sum_loss += example.w
            env.opt(self, example, idx, config, env)
        return sum_loss
    def reset(self, config, env):
        self.model.w = np.zeros(env.experiment.data_dim)
        self.acc_var = 1e-8

def search_clip_threshold(qs, la, lb, rc, rd):
    """
    assume qs is sorted, rc,la>=0
    return inf{M: Pr(q(X)<= la/M+lb) <= M*rc+rd }
    """
    n = len(qs)
    lo, hi = 0, n
    while lo<hi:
        mid = (lo+hi)//2
        cur_M = la/(qs[mid]-lb)
        if mid/n < cur_M*rc+rd:
            lo = mid+1
        else:
            hi = mid
    q = 1 if lo==n else qs[lo]
    #print("search clip: prop=%.2f q=%.5f th=%.2f bound=%.2f"%(lo/n, q, la/(q-lb), la/(q-lb)*rc+rd))
    return la/(q-lb)

def calc_clip_percentage(data, th):
    n_all = len(data)
    n_clipped = len([e for e in data if e.w>th])
    return n_clipped/n_all