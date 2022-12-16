##
# @file   ml_congestion.py
# @author Yibo Lin
# @date   Oct 2022
#

import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb

import dreamplace.ops.rudy.rudy as rudy
import dreamplace.ops.pinrudy.pinrudy as pinrudy
############## Your code block begins here ##############
from .gpdl import GPDL
############## Your code block ends here ################

class MLCongestion(nn.Module):
    """
    @brief compute congestion map based on a neural network model 
    @param fixed_node_map_op an operator to compute fixed macro map given node positions 
    @param rudy_utilization_map_op an operator to compute RUDY map given node positions
    @param pinrudy_utilization_map_op an operator to compute pin RUDY map given node positions 
    @param pin_pos_op an operator to compute pin positions given node positions 
    @param xl left boundary 
    @param yl bottom boundary 
    @param xh right boundary 
    @param yh top boundary 
    @param num_bins_x #bins in horizontal direction, assume to be the same as horizontal routing grids 
    @param num_bins_y #bins in vertical direction, assume to be the same as vertical routing grids 
    @param unit_horizontal_capacity amount of routing resources in horizontal direction in unit distance
    @param unit_vertical_capacity amount of routing resources in vertical direction in unit distance
    @param pretrained_ml_congestion_weight_file file path for pretrained weights of the machine learning model 
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 pretrained_ml_congestion_weight_file=None,
                 **kwargs):
        super(MLCongestion, self).__init__()
        ############## Your code block begins here ##############
        ############## Your code block ends here ################
        self.model = GPDL(in_channels=in_channels, out_channels=out_channels)
        self.model.eval()
        self.model.init_weights(pretrained=pretrained_ml_congestion_weight_file)
        self.model.cuda()

    def __call__(self, pos):
        return self.forward(pos)

    def forward(self, pos):  # note
        ############## Your code block begins here ##############
        return self.model.forward(pos)
        ############## Your code block ends here ################
