#
# Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Utility functions for converting and managing CrossSim Torch layers."""


import torch
import torch.nn as nn
import torch.nn.functional as F
SMOOTH_TW_WIDTH = 0.7
THRESHOLD = 0.05



# TNN — smooth threshold-window surrogate (w=0.7)
class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, threshold, width):
        ctx.save_for_backward(weight)
        ctx.threshold = threshold
        ctx.width = width
        return torch.where(
            weight > threshold,
            torch.ones_like(weight),
            torch.where(weight < -threshold, -torch.ones_like(weight), torch.zeros_like(weight)),
        )

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        threshold = ctx.threshold
        width = ctx.width
        dist_pos = (weight - threshold).abs()
        dist_neg = (weight + threshold).abs()
        grad_pos = torch.clamp(1.0 - dist_pos / width, min=0.0)
        grad_neg = torch.clamp(1.0 - dist_neg / width, min=0.0)
        grad_surrogate = torch.maximum(grad_pos, grad_neg)
        return grad_output * grad_surrogate, None, None


class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, threshold=THRESHOLD):
        super().__init__(in_features, out_features, bias)
        self.threshold = threshold

    def ternary_weight(self):
        return TernaryWeightFunction.apply(self.weight, self.threshold, SMOOTH_TW_WIDTH)

    def forward(self, input):
        return F.linear(input, self.ternary_weight(), self.bias)
