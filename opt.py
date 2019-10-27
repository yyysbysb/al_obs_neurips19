#!/usr/bin/env python3

import numpy as np
import math

def get_weight(model, dat, pred, eta):
    squared_norm = np.inner(dat.x, dat.x)
    exp = dat.w * eta * squared_norm
    if exp<1e-6:
        return 2 * (pred-dat.y) * dat.w * eta
    else:
        return (pred-dat.y) * (1.0 - np.exp(-2.0*exp))/squared_norm

def calc_gap(model, dat, cnt, eta):
    if model.w_accgrad is None:
        return abs(2*np.inner(model.w, dat.x) / (stepsize(cnt, eta)*np.inner(dat.x, dat.x)))
    else:
        tmp = sum(eta*dat.x*dat.x/np.sqrt(model.w_accgrad+1e-10))
        return abs(2*np.inner(model.w, dat.x) / tmp)

def stepsize(idx, stepsize_para0 = 0.05):
    return math.sqrt(stepsize_para0/(stepsize_para0+idx))

def gd(model, dat, idx, stepsize_para0 = 0.05):
    pred = model.predict(dat.x)
    model.w -= get_weight(model, dat, pred, stepsize(idx, stepsize_para0)) * dat.x

def get_weight_v2(dat, eta):
    squared_norm = np.inner(dat.x, dat.x)
    exp = dat.w * eta * squared_norm
    if exp<1e-6:
        return dat.w * eta
    else:
        return (1.0 - np.exp(-exp))/squared_norm

def gd_v2(learning, dat, idx, config, env):
    weight = get_weight_v2(dat, stepsize(idx, env.model.learning_rate))
    learning.model.w -= weight * learning.loss_grad(learning.model, dat)

def gd_var_v2(learning, dat, idx, config, env):
    var_coef = np.sqrt(env.model.c0)*dat.w*learning.loss_val(learning.model, dat)\
               /np.sqrt(env.learning.acc_var)
    learning.model.w -= get_weight_v2(dat, \
                          stepsize(idx, env.model.learning_rate)*(1+var_coef))\
                        * learning.loss_grad(learning.model, dat)

