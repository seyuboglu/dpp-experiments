"""
"""
import os
import sys
ROOT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_PATH, '..')) 

import torch
import numpy as np

from method.learned_cn_method import CNModule, VecCNModule


def test_simple():
    assert(0 == 0)