#!/usr/bin/env python3

import numpy as np

class CModel(object):
    def __init__(self, w=None, w_accgrad=None, acc_loss=0, reg_para=None, clip_para=None):
        self.w = w
        self.w_accgrad = w_accgrad
        self.acc_loss = acc_loss
        self.reg_para = reg_para
        self.clip_para = clip_para
    
    def predict(self, x):
        ret = np.inner(self.w, x)
        if ret>1: ret = 1
        if ret<-1: ret = -1
        return ret

def evaluate(model, data):
    if data is None or len(data)==0:
        return -1
    else:
        err= sum([1 if model.predict(dp.x)*dp.y<=0 else 0 for dp in data])
        return err/len(data)