#!/usr/bin/env python3

import copy
import json
import os
import utils
import numpy as np

class CConfig(object):
    def __init__(self, d):
        self.__dict__ = {k:utils.mycopy(v) for k,v in d.items()}

    def copy(self):
        return copy.deepcopy(self)

def dict2config(d):
    return None if d is None else CConfig({k:dict2config(v) if isinstance(v, dict) else v for k,v in d.items()})

def config2dict(config):
    if not hasattr(config, "__dict__"):
        return config
    else:
        return {k: config2dict(v) for k,v in config.__dict__.items()}

def load_config(config_paths, filename):
    config_path = None
    for path in config_paths:
        if os.path.isfile(path+filename):
            config_path = path+filename 
            break
    if config_path is not None:
        with open(config_path) as f:
            return dict2config(json.load(f))
    else:
        return None

def backup_config(config, path):
    with open(path, "w") as f:
        f.write(json.dumps(config2dict(config), sort_keys=True, indent=2))

class CEnvironment(object):
    def __init__(self, config=None, init_kv=None):
        if config is not None:
            self.__dict__ = {k:CEnvironment() for k in config.default_env_fields}
        elif init_kv is not None:
            self.__dict__ = {k:utils.mycopy(v) for k,v in init_kv.items()}
    def add_field(self, name):
        if not hasattr(self, name):
            setattr(self, name, CEnvironment())
    def remove_field(self, name):
        delattr(self, name)
    def prepare_for_multicore(self, config, idx):
        dataset = self.experiment.dataset
        self.experiment.remove_field("dataset")
        
        ret = self.copy()
        ret.remove_field("tmp")
        ret.experiment.dataset = dataset.copy()
        ret.experiment.random = np.random.RandomState(self.experiment.random.randint(100000))

        self.experiment.dataset = dataset
        return ret
    def copy(self):
        return copy.deepcopy(self)

