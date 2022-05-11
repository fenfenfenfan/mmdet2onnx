# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
from torch import nn
import torch.nn.functional as F 
from ...builder import HEADS
from typing import List
import spconv

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
@HEADS.register_module()
class RetinaNetHead_3x3(nn.Module):
    def __init__(self, num_classes, in_channels, conv_channels, num_convs, num_anchors):
        super().__init__()
        # fmt: off
        self.num_convs = num_convs
        # fmt: on

        self.cls_subnet = []
        self.bbox_subnet = []
        channels = in_channels
        for i in range(self.num_convs):
            cls_layer = nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1)
            bbox_layer = nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1)
            
            torch.nn.init.normal_(cls_layer.weight, mean=0, std=0.01)
            torch.nn.init.normal_(bbox_layer.weight, mean=0, std=0.01)
            
            torch.nn.init.constant_(cls_layer.bias, 0)
            torch.nn.init.constant_(bbox_layer.bias, 0)

            self.add_module('cls_layer_{}'.format(i), cls_layer)
            self.add_module('bbox_layer_{}'.format(i), bbox_layer)

            self.cls_subnet.append(cls_layer)
            self.bbox_subnet.append(bbox_layer)

            channels = conv_channels

        self.cls_score = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        torch.nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.01)

        torch.nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, features):
        logits = []
        bbox_reg = []

        for feature in features:
            cls_f  = feature
            bbox_f = feature 
            for i in range(self.num_convs):
                cls_f = F.relu(self.cls_subnet[i](cls_f))
                bbox_f = F.relu(self.bbox_subnet[i](bbox_f))

            logits.append(self.cls_score(cls_f))
            bbox_reg.append(self.bbox_pred(bbox_f))

        return logits, bbox_reg
    
    def get_params(self):
        cls_weights = [x.weight for x in self.cls_subnet] + [self.cls_score.weight.data]
        cls_biases = [x.bias for x in self.cls_subnet] + [self.cls_score.bias.data]

        bbox_weights = [x.weight for x in self.bbox_subnet] + [self.bbox_pred.weight.data]
        bbox_biases = [x.bias for x in self.bbox_subnet] + [self.bbox_pred.bias.data]
        return cls_weights, cls_biases, bbox_weights, bbox_biases
        
@HEADS.register_module()
class Head_3x3(nn.Module):
    def __init__(self, in_channels, conv_channels, num_convs, pred_channels, pred_prior=None):
        super().__init__()
        self.num_convs = num_convs

        self.subnet = []
        channels = in_channels
        for i in range(self.num_convs):
            layer = nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)
            self.add_module('layer_{}'.format(i), layer)
            self.subnet.append(layer)
            channels = conv_channels

        self.pred_net = nn.Conv2d(channels, pred_channels, kernel_size=3, stride=1, padding=1)

        torch.nn.init.xavier_normal_(self.pred_net.weight)
        torch.nn.init.constant_(self.pred_net.bias, 0)

    def forward(self, features):
        preds = []
        for feature in features:
            x = feature
            for i in range(self.num_convs):
                x = F.relu(self.subnet[i](x))
            preds.append(self.pred_net(x))
        return preds

    def get_params(self):
        weights = [x.weight for x in self.subnet] + [self.pred_net.weight]
        biases = [x.bias for x in self.subnet] + [self.pred_net.bias]
        return weights, biases

@HEADS.register_module()
class RetinaNetHead_3x3_MergeBN(nn.Module):
    def __init__(self, num_classes, in_channels, conv_channels, num_convs, num_anchors):
        super().__init__()
        # fmt: off
        num_anchors      = 1
        self.num_convs   = num_convs
        self.bn_converted = False
        # fmt: on

        self.cls_subnet = []
        self.bbox_subnet = []
        self.cls_bns = []
        self.bbox_bns = []

        channels = in_channels
        for i in range(self.num_convs):
            cls_layer = Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False, activation=None, norm=None)
            bbox_layer = Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False, activation=None, norm=None)
            torch.nn.init.normal_(cls_layer.weight, mean=0, std=0.01)
            torch.nn.init.normal_(bbox_layer.weight, mean=0, std=0.01)

            cls_bn = nn.SyncBatchNorm(conv_channels) 
            bbox_bn = nn.SyncBatchNorm(conv_channels)

            self.add_module('cls_layer_{}'.format(i), cls_layer)
            self.add_module('bbox_layer_{}'.format(i), bbox_layer)
            self.add_module('cls_bn_{}'.format(i), cls_bn)
            self.add_module('bbox_bn_{}'.format(i), bbox_bn)

            self.cls_subnet.append(cls_layer)
            self.bbox_subnet.append(bbox_layer)
            self.cls_bns.append(cls_bn)
            self.bbox_bns.append(bbox_bn)

            channels = conv_channels

        self.cls_score = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        torch.nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.01)

        torch.nn.init.constant_(self.cls_score.bias, 0)


    def forward(self, features, lvl_start):
        if self.training:
            return self._forward_train(features, lvl_start)
        else:
            return self._forward_eval(features, lvl_start)

    def _forward_train(self, features, lvl_start):
        cls_features = features
        bbox_features = features
        len_feats = len(features)

        for i in range(self.num_convs):          
            cls_features = [self.cls_subnet[i](x) for x in cls_features]
            bbox_features = [self.bbox_subnet[i](x) for x in bbox_features]

            cls_features = self.cls_bns[i](cls_features)
            bbox_features = self.bbox_bns[i](bbox_features)
            
            cls_features = [F.relu(x) for x in cls_features]
            bbox_features = [F.relu(x) for x in bbox_features]
        
        logits = [self.cls_score(x) for x in cls_features]
        bbox_pred = [self.bbox_pred(x) for x in bbox_features]
        return logits, bbox_pred
    

    def _forward_eval(self, features):
        if not self.bn_converted:
            self._bn_convert()
    
        cls_features = features
        bbox_features = features
        len_feats = len(features)

        for i in range(self.num_convs):
            cls_features = [F.relu(self.cls_subnet[i](x)) for x in cls_features]
            bbox_features = [F.relu(self.bbox_subnet[i](x)) for x in bbox_features]
        
        logits     = [self.cls_score(x) for x in cls_features]
        bbox_pred  = [self.bbox_pred(x) for x in bbox_features]

        return logits, bbox_pred

    def _bn_convert(self):
        # merge BN into head weights
        assert not self.training 
        if self.bn_converted:
            return

        for i in range(self.num_convs):
            cls_running_mean = self.cls_bns[i].running_mean.data
            cls_running_var = self.cls_bns[i].running_var.data
            cls_gamma = self.cls_bns[i].weight.data
            cls_beta  = self.cls_bns[i].bias.data 

            bbox_running_mean = self.bbox_bns[i].running_mean.data
            bbox_running_var = self.bbox_bns[i].running_var.data
            bbox_gamma = self.bbox_bns[i].weight.data
            bbox_beta  = self.bbox_bns[i].bias.data

            cls_bn_scale = cls_gamma * torch.rsqrt(cls_running_var + 1e-10)
            cls_bn_bias  = cls_beta - cls_bn_scale * cls_running_mean

            bbox_bn_scale = bbox_gamma * torch.rsqrt(bbox_running_var + 1e-10)
            bbox_bn_bias  = bbox_beta - bbox_bn_scale * bbox_running_mean

            self.cls_subnet[i].weight.data  = self.cls_subnet[i].weight.data * cls_bn_scale.view(-1, 1, 1, 1)
            self.cls_subnet[i].bias    = torch.nn.Parameter(cls_bn_bias)
            self.bbox_subnet[i].weight.data = self.bbox_subnet[i].weight.data * bbox_bn_scale.view(-1, 1, 1, 1)
            self.bbox_subnet[i].bias   = torch.nn.Parameter(bbox_bn_bias)

        self.bn_converted = True

    def get_params(self):
        if not self.bn_converted:
            self._bn_convert()

        cls_ws = [x.weight.data for x in self.cls_subnet] + [self.cls_score.weight.data]
        bbox_ws = [x.weight.data for x in self.bbox_subnet] + [self.bbox_pred.weight.data]

        cls_bs = [x.bias.data for x in self.cls_subnet] + [self.bbox_pred.weight.data]
        bbox_bs = [x.bias.data for x in self.bbox_subnet] + [self.bbox_pred.bias.data]

        return cls_ws, cls_bs, bbox_ws, bbox_bs

@HEADS.register_module()
class Head_3x3_MergeBN(nn.Module):
    def __init__(self, in_channels, conv_channels, num_convs, pred_channels, pred_prior=None):
        super().__init__()
        self.num_convs = num_convs
        self.bn_converted = False

        self.subnet = []
        self.bns    = []
        
        channels = in_channels
        for i in range(self.num_convs):
            layer = Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False, activation=None, norm=None)
            torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
            bn = nn.SyncBatchNorm(conv_channels) 

            self.add_module('layer_{}'.format(i), layer)
            self.add_module('bn_{}'.format(i), bn)

            self.subnet.append(layer)
            self.bns.append(bn)

            channels = conv_channels

        self.pred_net = nn.Conv2d(channels, pred_channels, kernel_size=3, stride=1, padding=1)
        
        torch.nn.init.normal_(self.pred_net.weight, mean=0, std=0.01)

        torch.nn.init.constant_(self.pred_net.bias, 0)

    def forward(self, features):
        if self.training:
            return self._forward_train(features)
        else:
            return self._forward_eval(features)
    
    def _forward_train(self, features):
        for i in range(self.num_convs):
            features = [self.subnet[i](x) for x in features]
            features = self.bns[i](features)
            features = [F.relu(x) for x in features]
        preds = [self.pred_net(x) for x in features]
        return preds
    
    def _forward_eval(self, features):
        if not self.bn_converted:
            self._bn_convert()

        for i in range(self.num_convs):
            features = [F.relu(self.subnet[i](x)) for x in features]
    
        preds = [self.pred_net(x) for x in features]
        return preds
    
    def _bn_convert(self):
        # merge BN into head weights
        assert not self.training 
        if self.bn_converted:
            return
        for i in range(self.num_convs):
            running_mean = self.bns[i].running_mean.data
            running_var = self.bns[i].running_var.data
            gamma = self.bns[i].weight.data
            beta  = self.bns[i].bias.data 
            bn_scale = gamma * torch.rsqrt(running_var + 1e-10)
            bn_bias  = beta - bn_scale * running_mean
            self.subnet[i].weight.data  = self.subnet[i].weight.data * bn_scale.view(-1, 1, 1, 1)
            self.subnet[i].bias    = torch.nn.Parameter(bn_bias)
        self.bn_converted = True

    def get_params(self):
        if not self.bn_converted:
            self._bn_convert()
        weights = [x.weight.data for x in self.subnet] + [self.pred_net.weight.data]
        biases  = [x.bias.data for x in self.subnet] + [self.pred_net.bias.data]
        return weights, biases



def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def run_conv2d(x, weights, bias):
    n_conv = len(weights)
    for i in range(n_conv):
        x = F.conv2d(x, weights[i], bias[i])
        if i != n_conv - 1:
            x = F.relu(x)
    return x


class QueryInfer(object):
    def __init__(self, anchor_num, num_classes, score_th=0.12, context=2):
        
        self.anchor_num  = anchor_num
        self.num_classes = num_classes
        self.score_th    = score_th
        self.context     = context 

        self.initialized = False
        self.cls_spconv  = None 
        self.bbox_spconv = None
        self.qcls_spconv = None
        self.qcls_conv   = None 
        self.n_conv      = None
    
    
    def _make_sparse_tensor(self, query_logits, last_ys, last_xs, anchors, feature_value):
        if last_ys is None:
            N, _, qh, qw = query_logits.size()
            assert N == 1
            prob  = torch.sigmoid_(query_logits).view(-1)
            pidxs = torch.where(prob > self.score_th)[0]# .float()
            y = torch.div(pidxs, qw).int()
            x = torch.remainder(pidxs, qw).int()
        else:
            prob  = torch.sigmoid_(query_logits).view(-1)
            pidxs = prob > self.score_th
            y = last_ys[pidxs]
            x = last_xs[pidxs]
        
        if y.size(0) == 0:
            return None, None, None, None, None, None 

        _, fc, fh, fw = feature_value.shape
        
        ys, xs = [], []
        for i in range(2):
            for j in range(2):
                ys.append(y * 2 + i)
                xs.append(x * 2 + j)

        ys = torch.cat(ys, dim=0)
        xs = torch.cat(xs, dim=0)
        inds = (ys * fw + xs).long()

        sparse_ys = []
        sparse_xs = []
        
        for i in range(-1*self.context, self.context+1):
            for j in range(-1*self.context, self.context+1):
                sparse_ys.append(ys+i)
                sparse_xs.append(xs+j)

        sparse_ys = torch.cat(sparse_ys, dim=0)
        sparse_xs = torch.cat(sparse_xs, dim=0)


        good_idx = (sparse_ys >= 0) & (sparse_ys < fh) & (sparse_xs >= 0)  & (sparse_xs < fw)
        sparse_ys = sparse_ys[good_idx]
        sparse_xs = sparse_xs[good_idx]
        
        sparse_yx = torch.stack((sparse_ys, sparse_xs), dim=0).t()
        sparse_yx = torch.unique(sparse_yx, sorted=False, dim=0)
        
        sparse_ys = sparse_yx[:, 0]
        sparse_xs = sparse_yx[:, 1]

        sparse_inds = (sparse_ys * fw + sparse_xs).long()

        sparse_features = feature_value.view(fc, -1).transpose(0, 1)[sparse_inds].view(-1, fc)
        sparse_indices  = torch.stack((torch.zeros_like(sparse_ys), sparse_ys, sparse_xs), dim=0).t().contiguous()
        
        sparse_tensor = spconv.SparseConvTensor(sparse_features, sparse_indices, (fh, fw), 1)
  
        anchors = anchors.tensor.view(-1, self.anchor_num, 4)
        selected_anchors = anchors[inds].view(1, -1, 4)
        return sparse_tensor, ys, xs, inds, selected_anchors, sparse_indices.size(0)

    def _make_spconv(self, weights, biases):
        nets = []
        for i in range(len(weights)):
            in_channel  = weights[i].shape[0]
            out_channel = weights[i].shape[1]
            k_size      = weights[i].shape[2]
            filter = spconv.SubMConv2d(in_channel, out_channel, k_size, 1, padding=k_size//2, indice_key="asd")
            filter.weight.data = weights[i].transpose(1,2).transpose(0,1).transpose(2,3).transpose(1,2).transpose(2,3)
            filter.bias.data   = biases[i]
            nets.append(filter)
            if i != len(weights) - 1:
                nets.append(torch.nn.ReLU())
        return spconv.SparseSequential(*nets)

    def _make_conv(self, weights, biases):
        nets = []
        for i in range(len(weights)):
            in_channel  = weights[i].shape[0]
            out_channel = weights[i].shape[1]
            k_size      = weights[i].shape[2]
            filter = torch.nn.Conv2d(in_channel, out_channel, k_size, 1, padding=k_size//2)
            filter.weight.data = weights[i]
            filter.bias.data   = biases[i]
            nets.append(filter)
            if i != len(weights) - 1:
                nets.append(torch.nn.ReLU())
        return torch.nn.Sequential(*nets)
    
    def _run_spconvs(self, x, filters):
        y = filters(x)
        return y.dense(channels_first=False)

    def _run_convs(self, x, filters):
        return filters(x)

    def run_qinfer(self, model_params, features_key, features_value, anchors_value):
        
        if not self.initialized:
            cls_weights, cls_biases, bbox_weights, bbox_biases, qcls_weights, qcls_biases = model_params
            assert len(cls_weights) == len(qcls_weights)
            self.n_conv = len(cls_weights)
            self.cls_spconv  = self._make_spconv(cls_weights, cls_biases)
            self.bbox_spconv = self._make_spconv(bbox_weights, bbox_biases)
            self.qcls_spconv = self._make_spconv(qcls_weights, qcls_biases)
            self.qcls_conv   = self._make_conv(qcls_weights, qcls_biases)
            self.initialized  = True

        last_ys, last_xs = None, None 
        query_logits = self._run_convs(features_key[-1], self.qcls_conv)
        det_cls_query, det_bbox_query, query_anchors = [], [], []
        
        n_inds_all = []

        for i in range(len(features_value)-1, -1, -1):
            x, last_ys, last_xs, inds, selected_anchors, n_inds = self._make_sparse_tensor(query_logits, last_ys, last_xs, anchors_value[i], features_value[i])
            n_inds_all.append(n_inds)
            if x == None:
                break
            cls_result   = self._run_spconvs(x, self.cls_spconv).view(-1, self.anchor_num*self.num_classes)[inds]
            bbox_result  = self._run_spconvs(x, self.bbox_spconv).view(-1, self.anchor_num*4)[inds]
            query_logits = self._run_spconvs(x, self.qcls_spconv).view(-1)[inds]
            
            query_anchors.append(selected_anchors)
            det_cls_query.append(torch.unsqueeze(cls_result, 0))
            det_bbox_query.append(torch.unsqueeze(bbox_result, 0))

        return det_cls_query, det_bbox_query, query_anchors


