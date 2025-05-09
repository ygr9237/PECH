import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, batch_size, device, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1))) 

    if batch_size == ad_out.size(0) // 2 :
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    else:
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * (ad_out.size(0) - batch_size))).float().to(device)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def DANN(features, ad_net,batch_size):
    ad_out = ad_net(features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size == ad_out.size(0) // 2 :
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    else:
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * (ad_out.size(0) - batch_size))).float().to(device)
    return nn.BCELoss()(ad_out, dc_target)
