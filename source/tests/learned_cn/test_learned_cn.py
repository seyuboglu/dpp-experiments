"""
"""
import os
import sys
ROOT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_PATH, '../..')) 

import torch
from torch import nn
import numpy as np

from method.learned_cn_method import CNModule, LCIModule
from util import Params, torch_all_close


class TestCN:

    def setup(self):
        """
        Builds vec_cn modules to test if they 
        """
        params = Params("source/tests/learned_cn/params.json")
        A = torch.tensor([[0., 1., 0.],
                          [1., 0., 1.],
                          [0., 1., 0.]])
        
        self.vec_cn = LCIModule(params, A.numpy())
    
    def test_1d(self):
        """
        Tests if the forward pass of the VecCN module accurately 
        reflects formulation. 
        """
        E = torch.tensor([1., 2., 3.]).view(1, 1, 3)
        W = torch.tensor([1., 3., 5.])
        X = torch.tensor([[1., 0., 1.],
                          [1., 1., 0.]])
        m, n = X.shape
        self.vec_cn.E = nn.Parameter(E)
        output = self.vec_cn(X)

        ### The correct output 
        correct = torch.tensor([[2.8284, 0.0000, 2.8284],
                                [1.4142, 2.0000, 1.4142]])
        ###

        correct = self.vec_cn.relu(correct)
        correct = self.vec_cn.linear(correct.view(self.vec_cn.d, m * n).t())
        correct = correct.view(m, n)

        assert(torch_all_close(correct, output, tolerance=1e-4))
    
    def test_2d(self):
        """
        Tests if the forward pass of the VecCH
        """
        pass 

