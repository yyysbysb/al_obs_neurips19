#!/usr/bin/env python3

import math
import tarfile
import glob
import os
import numpy as np

def mean(a):
    return sum(a)/len(a)

def stddev(a):
    if len(a)==1:
        return a[0]-a[0]
    m = mean(a)
    return np.sqrt(sum([(x-m)*(x-m) for x in a])*1.0/(len(a)-1))

def median(a):
    b = sorted(a)
    l = len(a)
    if l%2==0:
        return (b[l//2-1]+b[l//2])/2
    else:
        return b[l//2]

def robust_mean(a):
    b = sorted(a)
    l = len(b)//4
    r = len(b)-l
    if l<r:
        return mean(b[l:r])
    else:
        return mean(b)
def robust_stddev(a):
    b = sorted(a)
    l = len(b)//4
    r = len(b)-l
    if l<r:
        return stddev(b[l:r])
    else:
        return stddev(b)

def mycopy(a):
    return a.copy() if hasattr(a, "copy") else a
    
def my_float(s):
    try:
        return float(s)
    except ValueError:
        return 1e100

def my_dict_at(d, k):
    if d is None or k not in d:
        return None
    return d[k]
    
def textbf(filename):
    output = []
    with open(filename, "r") as f:
        for line in f:
            l = line.strip().split()
            minimum = min([my_float(s) for s in l]+[1e50])
            l_new = ["\\textbf{%s}"%minimum if my_float(s)==minimum else s for s in l]
            output.append(" ".join(l_new))
    with open(filename+".out", "w") as f:
        for l in output:
            f.write(l+"\n")

def backup_code(result_folder):
    with tarfile.open(result_folder+"code.tar", "w") as tar:
        for fn in glob.glob("*.py"):
            print(fn)
            tar.add(fn)
        for fn in glob.glob("*.json"):
            print(fn)
            tar.add(fn)
        for fn in glob.glob("AL-bandit/*.py"):
            print(fn)
            tar.add(fn)
        for fn in glob.glob("AL-bandit/*.json"):
            print(fn)
            tar.add(fn)


